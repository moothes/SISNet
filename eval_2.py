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
    parser.add_argument('--connect', default=True, help='Weight path of network')
    params = parser.parse_args()
    config = vars(params)
    
    config['orig_size'] = True
    config['data_path'] = '../dataset/'
    vals = ['PFOS', ]
    #print(vals)
    for val in vals:
        #img_path = '{}/{}/final/'.format(config['pre_path'], val) # our result
        img_path = config['pre_path'] # old methods
        if not os.path.exists(img_path):
            continue
        test_set = Test_Dataset(name=val, config=config)
        titer = test_set.size
        MR = MetricRecorder(titer)
        ious = []
        #MR = MetricRecorder()
        sim = []
        siou = []
        sgt = []
        
        counter = 0
        test_bar = Bar('Dataset {:10}:'.format(val), max=titer)
        for j in range(titer):
            _, gt, fdm, name = test_set.load_data(j)
            #new_name = '_'.join(name.split('/')) # our result
            new_name = name # old methods
            #print(new_name)
            pred = Image.open(img_path + new_name).convert('L')
            out_shape = gt.shape
            fdm = fdm.numpy()[0, 0]
            
            pred = np.array(pred.resize((out_shape[::-1])))
            
            thre, pred = cv2.threshold(pred * 255, 0, 255, cv2.THRESH_OTSU)
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
                #cv2.imwrite('temp/' + new_name, labels * 50)
                pred = new_pred
            MR.update(pre=pred, gt=gt.astype(np.float32))
            
            siou.append(cv2.resize(pred.astype(np.uint8), (320, 320)))
            sgt.append(cv2.resize(gt.astype(np.uint8), (320, 320)))
            
            counter += 1
            if counter == 15:
                #print(len(siou), len(sgt))
                #siou = cv2.resize(np.array(siou), (320, 320))
                #sgt = cv2.resize(np.array(sgt), (320, 320))
                #print(siou.shape, sgt.shape)
                si = np.array(siou)#.reshape(15, -1)
                sg = np.array(sgt)#.reshape(15, -1)
                
                si0 = si.reshape(15, 1, -1)
                si1 = si.reshape(15, -1, 1)
                sg0 = sg.reshape(15, 1, -1)
                sg1 = sg.reshape(15, -1, 1)
                
                g = 1 - sg0 * sg1
                inter = si0 * si1
                union = si0 + si1 - inter
                ious = np.sum(inter * g, axis=(1,2)) / np.sum(union, axis=(1,2))
                print(ious)
                siou = []
                sgt = []
                counter = 0
            
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