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
#from framework_factory import load_framework

def test_model(model, test_sets, config, epoch=None, saver=None):
    model.eval()
    if epoch is not None:
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}.pth'.format(config['model_name'], config['sub'], epoch))
        torch.save(model.state_dict(), weight_path)
    
    st = time.time()
    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = test_set.size
        MR = MetricRecorder(titer)
        #MR = MetricRecorder()
        ious = []
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, fdm, name = test_set.load_data(j)
            Y = model(image.cuda(), fdm.cuda())
            preds = Y['final'].sigmoid_().cpu().data.numpy()
            #preds = Y['fdm'].cpu().data.numpy()
            #preds = Y['final'].cpu().data.numpy()
            #print(preds.shape)
            pred = preds[0, 0]
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
            out_shape = gt.shape[-2:]
            
            pred = np.array(cv2.resize(pred, out_shape[::-1]))
            #pred = np.array(Image.fromarray(pred).resize((320, 320)))
            #gt = np.array(Image.fromarray(gt).resize((320, 320)))
            
            pred = (pred * 255).astype(np.uint8)
            thre, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
            pred, gt = normalize_pil(pred, gt)
            #pred = (pred > pred.mean()).astype(np.float32)
            
            #print(pred.shape, gt.shape)
            MR.update(pre=pred, gt=gt)
            #print(np.max(pred), np.max(gt))
            #MR.update(pre=(pred * 255).astype(np.uint8), gt=(gt * 255).astype(np.uint8))
            
            
            # save predictions
            if config['save']:
                tag = name.split('/')[0]
                fnl_folder = os.path.join(save_folder, 'final')
                check_path(fnl_folder)
                #im_path = os.path.join(fnl_folder, '_'.join(name.split('/')))
                #print(im_path)
                check_path(fnl_folder + '/' + tag)
                im_path = os.path.join(fnl_folder, name)
                cv2.imwrite(im_path, pred * 255)
                #Image.fromarray((pred * 255)).convert('L').save(im_path)
                #print(pred.shape)
                
                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                    pass
                
            inter = np.sum(pred * gt)
            union = np.sum(pred + gt) - inter
            iou = inter / (union + 1e-7)
            ious.append(iou)
            
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        #scores = MR.show(bit_num=3)
        #print('  Max-F: {}, adp-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(scores['fm'], scores['adpFm'], scores['wFm'], scores['MAE'], scores['Sm'], scores['adpEm']))
        #mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  Max-F: {}, Maen-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(maxf, meanf, wfm, mae, sm, em))
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  IOU: {}, Maen-F: {}, Fbw: {}, SM: {}, EM: {}.'.format(round(np.mean(ious), 3), meanf, wfm, sm, em))
        print('  IOU: {}, SM: {}, Fbw: {}, EM: {}, Mean-F: {}.'.format(round(np.mean(ious), 3), sm, wfm, em, meanf))
        
    print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    config['orig_size'] = False
    
    if config['weight'] != '':
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    #saved_model = torch.load(config['weight'], map_location='cpu')
    #new_name = {}
    #for k, v in saved_model.items():
    #    if k.startswith('model'):
    #        new_name[k[6:]] = v
    #    else:
    #        new_name[k] = v
    #model.load_state_dict(new_name)

    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    model = model.cuda()
    
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()