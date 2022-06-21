import sys
import importlib
from data import Test_Dataset
#from data_esod import ESOD_Test

import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np

from base.framework_factory import load_framework
from metric import *
from thop import profile
#from framework_factory import load_framework

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def test_model(model, test_sets, config, epoch=None, saver=None):
    model.eval()
    #time_count = 0
    
    #st = time.time()
    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = 1000 #test_set.size
        time_count = 0
        
        for j in range(titer):
            #image, gt, name = test_set.load_data(j)
            image, gt, fdm, name = test_set.load_data(j)
            image = image.cuda()
            fdm = fdm.cuda()
            
            torch.cuda.synchronize()
            st = time.time()
            Y = model(image, fdm)
            torch.cuda.synchronize()
            time_count += time.time() - st
            preds = Y['final'].sigmoid_().cpu().data.numpy()
            #pred = preds[0, 0] / (np.max(preds) + 1e-8)
            #out_shape = gt.shape
            
            #pred = np.array(Image.fromarray(pred).resize((out_shape[::-1])))
            
        #time_count = time.time() - st
        print('\nFPS: {}.'.format(titer / time_count))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)

    test_sets = {}
    test_sets['ESOD'] = Test_Dataset(name='ESOD', config=config)
    
    input = torch.randn(1, 3, config['size'], config['size']).cuda()
    input2 = torch.randn(1, 1, config['size'], config['size']).cuda()
    flops, _ = profile(model, inputs=(input, input2))
    params = params_count(model)
    print('FLOPs: {:.2f}, Params: {:.2f}.'.format(flops / 1e9, params / 1e6))
    
    #if not config['cpu']:
    model = model.cuda()
    
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()