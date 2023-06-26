from __future__ import print_function, division
import torch.nn.functional as F
import torch.nn as nn
import torch

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    
    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        
    Output:
        loss : dice loss of the epoch """

    bceloss = F.binary_cross_entropy_with_logits(prediction, target)
    diceprediction = torch.sigmoid(prediction)
    dice = dice_loss(diceprediction, target)
    
    loss = bceloss  + dice
    return loss

def corr_loss(prediction, target):
    vy = prediction - torch.mean(target)
    vx = target - torch.mean(target)
    corr = torch.sum(vx * vy, dim=0) * torch.rsqrt(torch.sum(vx ** 2, dim=0)) * torch.rsqrt(torch.sum(vy ** 2, dim=0))
    return  - torch.sum(corr**2)