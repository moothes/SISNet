import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()).cuda()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    #print(window)
    inter_img = img1 * img2
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    inter = F.conv2d(inter_img, window, padding = window_size//2, groups = channel)
    
    iou_map = 1 - (inter + 1e-7) / (mu1 + mu2 - inter + 1e-7)
    
    #print(float(torch.max(iou_map)), float(torch.min(iou_map)))
    #mu1_sq = mu1.pow(2)
    #mu2_sq = mu2.pow(2)
    #mu1_mu2 = mu1*mu2

    #sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    #sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    #sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    #C1 = 0.01**2
    #C2 = 0.03**2

    #ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return iou_map.mean()
    else:
        return iou_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel) # torch.ones((1, 1, window_size, window_size)).cuda() #
        #print(self.window.shape)

    def forward(self, img1, img2):
        inter_img = img1 * img2
        mu1 = F.conv2d(img1, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, self.window, padding = self.window_size//2, groups = self.channel)
        inter = F.conv2d(inter_img, self.window, padding = self.window_size//2, groups = self.channel)
        
        iou_map = 1 - (inter + 1e-7) / (mu1 + mu2 - inter + 1e-7)
        
        if self.size_average:
            return iou_map.mean()
        else:
            return iou_map.mean(1).mean(1).mean(1)
        #return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

def label_edge_prediction(label):
    ero = 1 - F.max_pool2d(1 - label, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(label, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge

def IOU(pred, target):
    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    iou_loss = 1 - (inter / union).mean()
    return iou_loss

def bce_ssim_loss(pred,target):
    bce_out = nn.BCELoss(size_average=True)(pred,target)
    #ssim_out = SSIM(window_size=11, size_average=True)(pred, target)
    iou_out = IOU(pred, target)
    loss = bce_out + iou_out #+ ssim_out
    return loss

def Loss(preds, target, fdm, config):
    loss = 0
    ws = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # [1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]
    wc = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    contour = label_edge_prediction(target)
    
    for pred, w in zip(preds['sal'], ws):
        #pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        tar_temp = F.interpolate(target, size=pred.size()[-2:], mode='bilinear')
        #print(bce_ssim_loss(torch.sigmoid(pred), tar_temp))
        loss += bce_ssim_loss(torch.sigmoid(pred), tar_temp) * w
        #loss += bce_ssim_loss(pred, target) * w
    
    ctr_loss = 0
    '''
    for pred, w in zip(preds['ctr'], wc):
        #pred = F.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        con_temp = F.adaptive_max_pool2d(contour, pred.size()[2:])
        ctr_loss += bce_ssim_loss(torch.sigmoid(pred), con_temp) * w
    '''
    
    bcg_loss = 0
    '''
    for pred in preds['bcg_map']:
        target = nn.functional.interpolate(target, size=pred.size()[-2:], mode='bilinear')
        fdm = nn.functional.interpolate(fdm, size=pred.size()[-2:], mode='bilinear')
        
        fdm_max = F.max_pool2d(fdm, kernel_size=5, stride=1, padding=2)
        new_fdm = (fdm_max == fdm).float() * fdm
        bce_out = F.binary_cross_entropy(pred, target, reduction='none')
        bcg_loss += torch.sum(bce_out * new_fdm) / torch.sum(new_fdm)
    '''
    
    #print(preds['fdm'].shape, target.shape)
    for pred in preds['fdm']:
        tar = target * fdm
        
        mask = fdm.gt(0.01).float()
        
        #print(target.shape, fdm.shape, pred.shape)
        
        #bcg_loss += torch.sum(torch.abs(tar - pred) * mask) / (torch.sum(mask) + 1e-5)
        #print(pred.shape, tar.shape)
        bcg_loss += torch.sum(F.binary_cross_entropy(pred, tar, reduction='none') * mask) / (torch.sum(mask) + 1e-5)
    
    #print(loss, bcg_loss)
    #print(ctr_loss)
    
    return loss + bcg_loss + ctr_loss