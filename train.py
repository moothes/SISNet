import sys
import os
import time
import random

#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import get_loader, Test_Dataset
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework

torch.set_printoptions(precision=5)

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    
    # Loading datasets
    train_loader = get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)
        
        #base_lr = (1-abs((epoch+1)/(config['epoch']+1)*2-1))*config['lr']

        print('Current lR: {}.'.format(optim.param_groups[-1]['lr']))
        st = time.time()
        loss_count = 0
        optim.zero_grad()
        sche.step()
        #print(optim.state_dict())
        for i, pack in enumerate(train_loader, start=1):
            '''
            it = i + (epoch-1) * num_iter
            warmup_steps = 100
            if it < warmup_steps:
                new_lr = config['lr'] * it / warmup_steps
            else:
                new_lr = config['lr'] * 0.9 * (1 - (it - warmup_steps) / (num_epoch * num_iter - warmup_steps)) + config['lr'] * 0.1
            optim.param_groups[0]['lr'] = new_lr * 0.1
            optim.param_groups[1]['lr'] = new_lr
            print(optim.param_groups[0]['lr'], optim.param_groups[1]['lr'])
            '''
            '''
            it = i + (epoch-1) * num_iter
            base_lr = config['lr']*((1 - it/(num_epoch * num_iter)) ** 0.9)
            optim.param_groups[0]['lr'] = base_lr*0.1
            optim.param_groups[1]['lr'] = base_lr
            '''
            
            images, gts, fdms = pack
            images, gts, fdms = images.float().cuda(), gts.float().cuda(), fdms.float().cuda()
            
            #print(torch.max(gts), torch.max(fdms))
            
            if config['multi']:
                scales = [-1, 0, 1] 
                #scales = [-2, -1, 0, 1, 2] 
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                #input_size += int(np.random.choice(scales, 1) * 32)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                fdms = F.upsample(fdms, size=(input_size, input_size), mode='nearest')
                
            Y = model(images, fdms, 'train')
            loss = model_loss(Y, gts, fdms, config) / ave_batch
            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            Bar.suffix = '{:4}/{:4} | loss: {:1.5f}, time: {}.'.format(i, num_iter, round(float(loss_count / i), 5), round(time.time() - st, 3))
            bar.next()

        bar.finish()
        
        test_model(model, test_sets, config, epoch)
            

if __name__ == "__main__":
    main()