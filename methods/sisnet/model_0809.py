import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout, up=True, d=1):
    yield nn.Conv2d(cin, cout, 3, padding=d, dilation=d)
    #yield nn.GroupNorm(cout // 2, cout)
    yield nn.GroupNorm(1, cout)
    #yield nn.BatchNorm2d(cout)
    #yield nn.Dropout(0.5)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')

def local_conv(cin, cout):
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(1, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Conv2d(cout, 1, 3, padding=1)

class info_block(nn.Module):
    def __init__(self, config, feat):
        super(info_block, self).__init__()
        self.res_conv1 = nn.Conv2d(feat, feat, 3, padding=1)
        self.fuse = nn.Conv2d(feat * 3, feat, 3, padding=1)
        #self.fdm_fuse = nn.Sequential(*list(up_conv(feat, feat, False)))
        self.sup = nn.Conv2d(feat, 1, 1)

    def forward(self, x0, x1, fdm):
        x0 = nn.functional.interpolate(x0, size=x1.size()[2:], mode='bilinear')
        fdm = nn.functional.interpolate(fdm, size=x1.size()[2:], mode='bilinear')
        #x1 = nn.functional.interpolate(x1, size=fdm.size()[2:], mode='bilinear')
        
        #fdm = self.fdm_fuse(fdm * x1)
        res_x1 = torch.sigmoid(self.res_conv1(x0 - x1))
        sup = self.sup(x0)
        res_x0 = torch.sigmoid(sup)
        x0 = self.fuse(torch.cat([x0 * res_x1, x1, x1 * res_x0], dim=1))
        #sup = self.sup(x0)
        
        return x0, sup, None
        
class fdm_refine_block(nn.Module):
    def __init__(self, config, feat):
        super(fdm_refine_block, self).__init__()
        #self.conv1 = nn.Sequential(*list(up_conv(1, 8, False)))
        #self.conv2 = nn.Sequential(*list(up_conv(8, 8, False)))
        
        #self.sup = nn.Conv2d(8, 1, 1)

    def forward(self, fdmx, x):
        #center = F.adaptive_avg_pool2d(x, (1, 1))
        #map = torch.sum(torch.abs(center - x), dim=1, keepdim=True)
        #map = nn.functional.interpolate(map, size=fdmx.size()[2:], mode='bilinear')
        
        #fdm = self.conv1(fdmx)
        #fdm = self.conv2(fdm)
        
        #sup = torch.sigmoid(map) #* fdm
        #fdm = fdmx * sup
        return fdmx, x

class fdm_expand_block(nn.Module):
    def __init__(self, config, feat, ds=[7, 5, 3, 1]):
        super(fdm_expand_block, self).__init__()
        #self.conv1 = nn.Sequential(*list(up_conv(feat * 2, feat, False)))
        self.conv0 = nn.Sequential(*list(up_conv(1, feat, False, d=1)))
        #self.conv0_1 = nn.Sequential(*list(up_conv(feat, feat, False, d=1)))
        self.conv1 = nn.Sequential(*list(up_conv(feat, feat, False, d=ds[0])))
        self.conv2 = nn.Sequential(*list(up_conv(feat, feat, False, d=ds[1])))
        self.conv3 = nn.Sequential(*list(up_conv(feat, feat, False, d=ds[2])))
        self.conv4 = nn.Sequential(*list(up_conv(feat, feat, False, d=ds[3])))
        self.fuse_conv1 = nn.Sequential(*list(up_conv(feat * 4, feat, False)))
        self.fuse_conv2 = nn.Sequential(*list(up_conv(feat, feat, False)))
        
        self.sup = nn.Conv2d(feat, 1, 1)
        
        self.fuse = nn.Sequential(*list(up_conv(feat * 2, feat, False)))

    def forward(self, fdm, x):
        '''
        x_size = x.size()[-2:]
        fdm = F.max_pool2d(fdm, kernel_size=9, stride=8, padding=4)
        x0 = nn.functional.interpolate(x, size=fdm.size()[2:], mode='bilinear')
        x_atten = fdm * x0
        b, c, w, h = x_atten.size()
        
        x_fla = x0.view(b, c, 1, w * h)
        x_atten_fla = x_atten.view(b, c, w * h, 1)
        
        score_map = torch.sum(x_fla * x_atten_fla, dim=1)
        score_map = torch.max(score_map, dim=1)[0]
        score_map = score_map.view(b, 1, w, h)
        score_map = nn.functional.interpolate(score_map, size=x_size, mode='bilinear')
        #print(score_map.shape)
        '''
        
        fdm = F.max_pool2d(fdm, kernel_size=13, stride=1, padding=6)
        fdm = nn.functional.interpolate(fdm, size=x.size()[2:], mode='bilinear')
        fdm = self.conv0(fdm)
        #fdm = self.conv0_1(fdm)
        
        
        
        x1 = self.conv1(fdm)
        x2 = self.conv2(fdm)
        x3 = self.conv3(fdm)
        x4 = self.conv4(fdm)
        
        xfuse = torch.cat([x1, x2, x3, x4], dim=1)
        xfuse = self.fuse_conv1(xfuse)
        xfuse = self.fuse_conv2(xfuse)
        
        #x1 = self.conv1(torch.cat([x, x * score_map], dim=1))
        #x1 = self.conv1(fdm)
        #print(x1.shape)
        #x1 = self.conv2(x1)
        #print(x1.shape)
        #x1 = self.conv3(x1)
        #print(x1.shape)
        #x1 = self.conv4(x1)
        #print(x1.shape)
        
        sup = self.sup(xfuse)
        x0 = self.fuse(torch.cat([xfuse * x, x], dim=1))
        return x0, sup



class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        num_channel = feat[0] # 128 # feat[0]
        
        #self.adapter = [nn.Sequential(*list(up_conv(feat[i], feat[0], False))).cuda() for i in range(5)]
        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], num_channel, False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], num_channel, False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], num_channel, False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], num_channel, False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], num_channel, False)))
        
        self.fdm_ref = fdm_refine_block(config, num_channel)
        
        self.fdm_exp0 = fdm_expand_block(config, num_channel, [3, 5, 7, 9])
        self.fdm_exp1 = fdm_expand_block(config, num_channel, [2, 4, 6, 8])
        self.fdm_exp2 = fdm_expand_block(config, num_channel, [1, 3, 5, 7])
        self.fdm_exp3 = fdm_expand_block(config, num_channel, [1, 2, 4, 6])
        self.fdm_exp4 = fdm_expand_block(config, num_channel, [1, 2, 3, 4])
        
        self.info0 = info_block(config, num_channel)
        self.info1 = info_block(config, num_channel)
        self.info2 = info_block(config, num_channel)
        self.info3 = info_block(config, num_channel)
        self.info4 = info_block(config, num_channel)
        
        
        self.final_pred = nn.Conv2d(num_channel, 1, 1)
        #self.final_pred = nn.Sequential(*list(local_conv(num_channel, num_channel)))
        
        
    def forward(self, xs, fdm, x_size):
        #xs = [self.adapter[i](xs[i]) for i in range(5)]
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])
        
        #fdm = nn.functional.interpolate(fdm, size=xs[0].size()[2:], mode='bilinear')
        fdm = F.adaptive_max_pool2d(fdm, xs[0].size()[2:])
        fdm, fdm0 = self.fdm_ref(fdm, xs[4])
        
        
        x4, sup4 = self.fdm_exp0(fdm, xs[4])
        x3, sup3 = self.fdm_exp1(fdm, xs[3])
        x2, sup2 = self.fdm_exp2(fdm, xs[2])
        x1, sup1 = self.fdm_exp3(fdm, xs[1])
        x0, sup0 = self.fdm_exp4(fdm, xs[0])
        
        p, sup_4, bup_4 = self.info0(x4, x4, fdm)
        p, sup_3, bup_3 = self.info1(p, x3, fdm)
        p, sup_2, bup_2 = self.info2(p, x2, fdm)
        p, sup_1, bup_1 = self.info3(p, x1, fdm)
        p, sup_0, bup_0 = self.info4(p, x0, fdm)
        pred = self.final_pred(p)
        #print(pred.shape)

        #print(torch.max(sup0))
        OutDict = {}
        #OutDict['atten'] = [r1, r2, r3, r4]
        #OutDict['feat'] = [loc_x, reg_x]
        OutDict['fdm'] = [fdm0, ]
        OutDict['ctr'] = [bup_0, bup_1, bup_2, bup_3, bup_4]
        OutDict['sal'] = [pred, sup0, sup1, sup2, sup3, sup4, sup_0, sup_1, sup_2, sup_3, sup_4]
        OutDict['final'] = pred

        return OutDict
        

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, fdm, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, fdm, x_size)
        return out
