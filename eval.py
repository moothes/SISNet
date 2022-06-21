import sys
import importlib
from data import Test_Dataset
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
import argparse

from skimage import measure, color
from metric import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_path', default='./maps', help='Weight path of network')
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--trset', default='PFOS')
    parser.add_argument('--size', default=320)
    params = parser.parse_args()
    config = vars(params)
    
    config['orig_size'] = True
    config['data_path'] = '../dataset/'
    vals = config['trset'] #['PFOS', ] # 'OSIE-CFPS', 
    #print(vals)
    for val in vals:
        #img_path = '{}/{}/final/'.format(config['pre_path'], val)
        img_path = config['pre_path']
        if not os.path.exists(img_path):
            continue
        test_set = Test_Dataset(name=val, config=config)
        titer = test_set.size
        MR = MetricRecorder(titer)
        ious = []
        #MR = MetricRecorder()
        
        test_bar = Bar('Dataset {:10}:'.format(val), max=titer)
        for j in range(titer):
            _, gt, fdm, name = test_set.load_data(j)
            #new_name = '_'.join(name.split('/'))
            new_name = name
            #print(new_name)
            pred = Image.open(img_path + new_name).convert('L')
            out_shape = gt.shape
            fdm = fdm.numpy()[0, 0]
            
            pred = np.array(pred.resize((out_shape[::-1])))
            thre, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
            
            pred, gt = normalize_pil(pred, gt)
            if config['connect']:
                labels = measure.label(pred, connectivity=2)
                lbls = np.unique(labels)
                new_pred = np.zeros_like(pred)
                for lbl in lbls:
                    if lbl == 0:
                        continue
                    region = (labels == lbl).astype(np.float32)
                    amap = region * fdm
                    if np.max(amap) > 0.01:
                        new_pred += region
                #print(labels.shape, fdm.shape)
                cv2.imwrite('temp/' + new_name, labels * 50)
                pred = new_pred
            MR.update(pre=pred, gt=gt.astype(np.float32))
            
            inter = np.sum(pred * gt)
            union = np.sum(pred + gt) - inter
            iou = inter / (union + 1e-7)
            ious.append(iou)
            
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        #scores = MR.show(bit_num=3)
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print('  IOU: {}, Maen-F: {}, Fbw: {}, SM: {}, EM: {}.'.format(round(np.mean(ious), 3), meanf, wfm, sm, em))
        #mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))

    
if __name__ == "__main__":
    main()