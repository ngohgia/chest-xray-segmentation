import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys

class BCEWithLogitLoss2D(nn.Module):
    # Binary Cross Entropy + 2D Sigmoid
    # TODO: focal loss instead of weight
    # TODO: weights for the scores
    
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitLoss2D, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)
        
    def forward(self, scores, targets):
        flat_scores = scores.view(-1)
        flat_targets = targets.view(-1)
        return self.loss(flat_scores, flat_targets)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, scores, targets):
        scores = torch.sigmoid(scores)
        flat_scores = scores.view(-1)
        flat_targets = targets.view(-1)
        intersection = (flat_scores * flat_targets).sum()
    
        return 1 - ((2. * intersection + self.smooth) /
              (flat_scores.sum() + flat_targets.sum() + self.smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
