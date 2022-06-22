import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *

def CTLoss(preds, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 4 + 1
    loss = (bce(preds, target) * wm).mean()
    return loss
    
    
def Fscore(preds, target, config):
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 0.8 + 0.2
    pred = torch.sigmoid(preds)
    tp = wm * pred * target
    pred = wm * pred
    target = wm * target
    
    fs = 1.3 * tp.sum(dim=(1, 2, 3)) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) * 0.3)
    loss = 1 - fs.mean()
    
    return loss

def IOU(pred, target):
    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    iou_loss = 1 - (inter / union).mean()
    return iou_loss

def cff_loss(preds, target, config):
    c = CTLoss(preds, target, config)
    f = Fscore(preds, target, config)
    
    return c + 2 * f

def Loss_new(X, target, config):
    bce_loss = cff_loss(X['final'], target.gt(0.5).float(), config)

    return bce_loss


def Loss(preds, target, fdm, config):
    bce = nn.BCEWithLogitsLoss()
    
    loss = 0
    for pred in preds['sal']:
        loss += bce(pred, target.gt(0.5).float())

        #loss += IOU()
        p = torch.sigmoid(pred)
        intersection = p * target
        numerator = (p - intersection).sum() + (target - intersection).sum()
        denominator = p.sum() + target.sum()
        loss += numerator / (denominator + 1e-6)
    return loss

