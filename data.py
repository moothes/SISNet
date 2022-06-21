import os, glob, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import random
import time
from progress.bar import Bar
import cv2
from scipy.io import loadmat


#mean = np.array((104.00699, 116.66877, 122.67892)).reshape((1, 1, 3))
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def get_image_list(config, phase):
    images = []
    gts = []
    fdms = []
    fdps = []
    
    image_root = os.path.join(config['data_path'], config['trset'], phase, 'JPEGimages')
    gt_root = os.path.join(config['data_path'], config['trset'], phase, 'GT')
    fdm_root = os.path.join(config['data_path'], config['trset'], phase, 'FDM')
    fdp_root = os.path.join(config['data_path'], config['trset'], phase, 'fixations')
    
    for fold in os.listdir(gt_root):
        fold_path = os.path.join(gt_root, fold)
        img_list = os.listdir(fold_path)
        for img_name in img_list:
            images.append(os.path.join(image_root, fold + '.jpg'))
            gts.append(os.path.join(gt_root, fold, img_name))
            fdms.append(os.path.join(fdm_root, fold, img_name))
            fdps.append(os.path.join(fdp_root, fold, img_name.split('.')[0] + '.mat'))
            #print(os.path.join(fold_path, img_name))
    
    #images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
    #gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
    #print(len(images), len(gts))
    
    return images, gts, fdms, fdps

def get_loader(config):
    dataset = Train_Dataset(config['trset'], config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def get_fdm(fdps, orig_size, new_size):
    fdps = fdps[:, 1::-1]
    ratios = np.expand_dims(new_size / np.array(orig_size), axis=0)
    fdps = (ratios * fdps).astype(np.int32)
    fdm = np.zeros((1, new_size, new_size))
    for fdp in fdps:
        fdm[0, fdp[0], fdp[1]] = 1
    #if np.max(fdps) > 320:
    #    print(fdps)
    return fdm

class Train_Dataset(data.Dataset):
    def __init__(self, name, config):
        self.config = config
        self.images, self.gts, self.fdms, self.fdps = get_image_list(config, 'train')
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        fdm = Image.open(self.fdms[index]).convert('L')
        
        #m = loadmat(self.fdps[index])
        #fdm = get_fdm(m['fixations'], np.array(gt).shape, self.config['size'])
        
        if self.config['data_aug']:
            image, gt = rotate(image, gt)
            image = random_light(image)
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size))
        fdm = fdm.resize((img_size, img_size))
    
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        fdm = np.array(fdm)
        
        #print(image.shape, gt.shape)
        if random.random() > 0.5:
            image = image[:, ::-1]
            gt = gt[:, ::-1]
            fdm = fdm[:, ::-1]
            
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims((gt > 128).astype(np.float32), axis=0)
        #fdm = np.ascontiguousarray(fdm, dtype=np.float32)
        fdm = np.expand_dims(fdm / 255., axis=0)
        #fdm = torch.tensor(fdm).float().unsqueeze(0)
        #fdm = np.expand_dims((fdm > 128).astype(np.float32), axis=0)
        #gt = gt / 255.
        #if not os.path.exists('./temp' + self.gts[index][-14:-9]):
        #    os.mkdir('./temp' + self.gts[index][-14:-9])
        #cv2.imwrite('./temp' + self.gts[index][-14:], fdm[0] * 255)
        #print('./temp' + self.gts[index][-14:])
        return image, gt, fdm

    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts, self.fdms, self.fdps = get_image_list(config, 'test')
        self.size = len(self.images)

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if not self.config['orig_size']:
            image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        fdm = np.array(Image.open(self.fdms[index]).convert('L').resize((self.config['size'], self.config['size'])))
        name = '/'.join(self.gts[index].split('/')[-2:])
        
        #m = loadmat(self.fdps[index])
        #fdm = get_fdm(m['fixations'], np.array(gt).shape, self.config['size'])
        #print(name)
        #print(self.gts[index], name)
        
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.tensor(np.expand_dims(image, 0)).float()
        #gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        gt = (gt > 128).astype(np.float32)
        #fdm = np.expand_dims((fdm > 128).astype(np.float32), axis=0)
        fdm = np.expand_dims(fdm / 255., axis=0)
        fdm = torch.tensor(fdm).float().unsqueeze(0)
        #gt = (gt > 0.5).astype(np.float32)
        #gt = (gt > 128).astype(np.float32)
        #print(image.shape, gt.shape, fdm.shape)
        return image, gt, fdm, name

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()