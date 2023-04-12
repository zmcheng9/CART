"""
CART fot FSMIS
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
from models.MCAB import MCA
from focal_loss.focal_loss import FocalLoss


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
        self.criterion_focal = FocalLoss(gamma=0.7, weights=self.my_weight)
        self.alpha = torch.Tensor([1.0, 0.0])
        self.m = 6
        self.MaskMAB = MCA(self.m, 512, 512, 512*self.m)
        self.k = 10


    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    # 1
        supp_bs = supp_imgs[0][0].shape[0]      # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        ##### Generate adpative threshold #######
        self.thresh_pred = [tao[:self.n_ways * self.n_shots * supp_bs], tao[self.n_ways * self.n_shots * supp_bs:]]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        sup_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            # if supp_mask[epi][0].sum() == 0:
            supp_fts_ = [[self.getFeatures(supp_fts[0][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]  # every fig prototype
            fg_prototypes = self.getPrototype(supp_fts_)  # the prototypes for support images

            ###### Get query predictions ######
            qry_preds = []
            sup_preds = []
            for way in range(self.n_ways):
                qry_pred, sup_pred = self.CART(supp_fts[0][[epi], way, 0], qry_fts[0][epi], fg_prototypes[way],
                                               self.k, self.thresh_pred[1])
                qry_preds.append(qry_pred)
                sup_preds.append(sup_pred)
            qry_preds = torch.stack(qry_preds, dim=1)
            sup_preds = torch.stack(sup_preds, dim=1)

            qry_preds = F.interpolate(qry_preds, size=img_size, mode='bilinear', align_corners=True)
            sup_preds = F.interpolate(sup_preds, size=img_size, mode='bilinear', align_corners=True)

            qry_preds = torch.cat((1.0 - qry_preds, qry_preds), dim=1)
            outputs.append(qry_preds)

            sup_preds = torch.cat((1.0 - sup_preds, sup_preds), dim=1)

            ''' Prototype alignment loss '''
            if train:
                align_loss_epi = self.alignLoss([supp_fts[0][epi]], [qry_fts[0][epi]],
                                                qry_preds, supp_mask[epi])
                align_loss += align_loss_epi

                sup_loss_epi = self.sup_loss(sup_preds, supp_mask[epi])
                sup_loss += sup_loss_epi


        outputs = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        outputs = outputs.view(-1, *outputs.shape[2:])

        return outputs, align_loss / supp_bs, sup_loss / supp_bs


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512), (1, 1)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))   # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)  
        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts


        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        pred_mask = torch.sum(pred_mask, dim=0)
        pred_mask = torch.where(pred_mask >= ((self.k+1) / 4), 1, 0).unsqueeze(0)  # 1 x h x w
        # pred_mask = torch.where(pred_mask >= ((self.k+1) / 4), 1, 0).unsqueeze(0)    # 1 x h x w
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[0])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True) for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def CART(self, sup_fts, qry_fts, prototype, iters, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512, 64, 64) (1, 512) 3
        Args:
            sup_fts: input features
                expect shape: 1 x 512 x H x W
            qry_fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        temp_fts = qry_fts
        prototype_q = prototype
        qry_pts = []
        sup_pts = []
        for i in range(iters):
            prototype_q = self.MaskMAB(prototype_q, temp_fts) + prototype_q
            if i % 2 == 0:
                qry_pts.append(prototype_q)
                temp_fts = sup_fts
            else:
                sup_pts.append(prototype_q)
                temp_fts = qry_fts

        qry_pts = torch.cat(qry_pts, dim=0)
        sup_pts = torch.cat(sup_pts, dim=0)
        pred = self.getPred(qry_fts, qry_pts, thresh)
        sup_pred = self.getPred(sup_fts, sup_pts, self.thresh_pred[0])

        return pred, sup_pred


    def sup_loss(self, sup_preds, fore_mask): 

        # Construct the support Ground-Truth segmentation
        supp_label = torch.full_like(fore_mask[0, 0], 255, device=fore_mask.device)
        supp_label[fore_mask[0, 0] == 1] = 1
        supp_label[fore_mask[0, 0] == 0] = 0

        loss = torch.zeros(1).to(self.device)
        for i in range(len(sup_preds)):
            # Compute sup_loss
            loss += self.criterion_focal(sup_preds[i].unsqueeze(0).permute(0, 2, 3, 1), supp_label[None, ...].long())

        return loss / len(sup_preds)






