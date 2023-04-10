import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
         return nn.CrossEntropyLoss(ignore_index=255, reduction='elementwise_mean')#reduction='mean'
    elif loss_type =='DiceLoss':
        return DiceLoss()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def cross_entropy_loss(prediction, label):
            # print (label,label.max(),label.min())
            label = label.long()
            mask = (label != 0).float()
            num_positive = torch.sum(mask).float()
            num_negative = mask.numel() - num_positive
            # print (num_positive, num_negative)
            mask[mask != 0] = num_negative / (num_positive + num_negative)
            mask[mask == 0] = num_positive / (num_positive + num_negative)

            ce_loss = F.cross_entropy(prediction, label,reduction='none',weight=mask, ignore_index=255)

            return torch.sum(ce_loss)

def cross_entropy_loss_b(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        # print(input.size(),target.size())
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss


def bce2d_new(input, target):
    print(input.size(),target.size())
    # input = input.squeeze(1)
    # target = target.squeeze(1)
    b,c,h, w = input.size()
    # assert (input.size() == target.size())
    b_gt = np.zeros(( b,c,h, w))
    # # b_i = np.zeros((b, 1, h, w))
    # for i in range(b):
    #     y = target[i].cpu().numpy()
    #     y = y.astype(np.uint8)
    #     y = y * 255
    #     print(y.max())
    #     plt.imshow(y)
    #     plt.show()
    #     lin=cv2.Canny(y,10,100)
    #     plt.imshow(b_gt[i])
    #     plt.show()
    #     # lin =lin.unsqueeze(1)
    #     b_gt[i][lin==1]=1
    #     plt.imshow(b_gt[i])
    #     plt.show()


    # b_gt = torch.from_numpy(b_gt).cuda().float()

    b_gt[target==0]=1
    # print(b_gt.max())
    # b_i = torch.from_numpy(b_i).cuda().float()
    pos = torch.eq(b_gt, 1).float()
    neg = torch.eq(b_gt, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, b_gt, weights, size_average=True)
