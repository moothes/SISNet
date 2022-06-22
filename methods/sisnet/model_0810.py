import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout, d=1):
    yield nn.Conv2d(cin, cout, 3, padding=d, dilation=d)
    #yield nn.GroupNorm(cout // 2, cout)
    yield nn.GroupNorm(1, cout)
    #yield nn.BatchNorm2d(cout)
    #yield nn.Dropout(0.5)
    yield nn.ReLU(inplace=True)

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
        self.fuse = nn.Sequential(*list(up_conv(feat * 3, feat)))
        #self.conv2 = nn.Sequential(*list(up_conv(8, 8, False)))
        self.bcg_conv = nn.Conv2d(feat, 1, 1)
        #self.sup = nn.Conv2d(8, 1, 1)

    def forward(self, fdm, xs):
        
        x0 = F.interpolate(xs[0], size=fdm.shape[2:], mode='bilinear')
        x1 = F.interpolate(xs[1], size=fdm.shape[2:], mode='bilinear')
        x2 = F.interpolate(xs[2], size=fdm.shape[2:], mode='bilinear')
        
        base_feat = self.fuse(torch.cat([x0, x1, x2], dim=1))
        bcg_map = self.bcg_conv(base_feat)
        
        bmap = torch.sigmoid(bcg_map)
        mean_bcg = torch.mean(bmap, dim=(2,3), keepdim=True)
        fdm = fdm * (bmap > mean_bcg).float()
        
        return fdm, bcg_map

class fdm_expand_block(nn.Module):
    def __init__(self, config, feat, ds=[7, 5, 3, 1], k=5):
        super(fdm_expand_block, self).__init__()
        #self.conv1 = nn.Sequential(*list(up_conv(feat * 2, feat, False)))
        self.conv0 = nn.Sequential(*list(up_conv(1, feat, d=1)))
        #self.conv0_1 = nn.Sequential(*list(up_conv(feat, feat, False, d=1)))
        self.conv1 = nn.Sequential(*list(up_conv(feat, feat, d=ds[0])))
        self.conv2 = nn.Sequential(*list(up_conv(feat, feat, d=ds[1])))
        self.conv3 = nn.Sequential(*list(up_conv(feat, feat, d=ds[2])))
        self.conv4 = nn.Sequential(*list(up_conv(feat, feat, d=ds[3])))
        self.fuse_conv1 = nn.Sequential(*list(up_conv(feat * 4, feat)))
        self.fuse_conv2 = nn.Sequential(*list(up_conv(feat, feat)))
        
        self.sup = nn.Conv2d(feat, 1, 1)
        
        self.fuse = nn.Sequential(*list(up_conv(feat * 2, feat)))
        self.k = k

    def forward(self, fdm, x):
        
        fdm = F.max_pool2d(fdm, kernel_size=self.k, stride=1, padding=(self.k - 1) // 2)
        fdm = F.avg_pool2d(fdm, kernel_size=self.k, stride=1, padding=(self.k - 1) // 2)
        fdm = nn.functional.interpolate(fdm, size=x.size()[2:], mode='bilinear')
        fdm = self.conv0(fdm)
        
        x1 = self.conv1(fdm)
        x2 = self.conv2(fdm)
        x3 = self.conv3(fdm)
        x4 = self.conv4(fdm)
        
        xfuse = torch.cat([x1, x2, x3, x4], dim=1)
        xfuse = self.fuse_conv1(xfuse)
        xfuse = self.fuse_conv2(xfuse)
        
        sup = self.sup(xfuse)
        x0 = self.fuse(torch.cat([xfuse * x, x], dim=1))
        return x0, sup

class fdm_expand_block_new(nn.Module):
    def __init__(self, config, feat, ds=[7, 5, 3, 1]):
        super(fdm_expand_block_new, self).__init__()
        self.conv1 = nn.Sequential(*list(up_conv(feat * 2, feat)))
        self.conv0 = nn.Sequential(*list(up_conv(1, feat, d=1)))
        #self.conv0_1 = nn.Sequential(*list(up_conv(feat, feat, d=1)))
        self.conv1 = nn.Sequential(*list(up_conv(feat, feat, d=ds[0])))
        self.conv2 = nn.Sequential(*list(up_conv(feat, feat, d=ds[1])))
        self.conv3 = nn.Sequential(*list(up_conv(feat, feat, d=ds[2])))
        self.conv4 = nn.Sequential(*list(up_conv(feat, feat, d=ds[3])))
        self.fuse_conv1 = nn.Sequential(*list(up_conv(feat * 4, feat)))
        self.fuse_conv2 = nn.Sequential(*list(up_conv(feat, feat, False)))
        
        self.sup = nn.Conv2d(feat, 1, 1)
        
        self.fuse = nn.Sequential(*list(up_conv(feat * 2, feat)))

    def forward(self, fdm, x):
        #x1 = self.conv1(x)
        #x2 = self.conv2(x)
        #x3 = self.conv3(x)
        #x4 = self.conv4(x)
        
        #xfuse = torch.cat([x1, x2, x3, x4], dim=1)
        #xfuse = self.fuse_conv1(xfuse)
        xfuse = x
        
        fdm = F.adaptive_max_pool2d(fdm, xfuse.size()[2:])
        #fdm = nn.functional.interpolate(fdm, size=x.size()[2:], mode='bilinear')
        #fdm = (fdm > thre).bool()
        
        #b = fdm.size(0)
        b, f, w, h = xfuse.size()
        sim_maps = []
        for i in range(b):
            x_temp = xfuse[i].permute(1, 2, 0).contiguous()
            fdm_temp = fdm[i].permute(1, 2, 0).contiguous()
            
            thre = min(0.5, torch.max(fdm_temp) / 3.)
            #if thre < 0.5:
            #    print(thre)
            fdm_temp = (fdm_temp >= thre).bool()
            select_feat = torch.masked_select(x_temp, fdm_temp)
            select_feat = select_feat.view(-1, 1, f)
            
            
            x_flat = x_temp.view(1, -1, f)
            sim_map = F.cosine_similarity(select_feat, x_flat, dim=-1) / 2. + 0.5
            #print(sim_map.shape)
            sim_map = torch.max(sim_map, dim=0)[0].view(1, w, h)
            sim_maps.append(sim_map)
        
        sm = torch.stack(sim_maps, dim=0)
        #print(sm.shape)
        
        #x0 = self.fuse(torch.cat([xfuse * x, x], dim=1))
        x0 = self.fuse(torch.cat([sm * x, x], dim=1))
        return x0, sm



class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        num_channel = feat[0] # 128 # feat[0]
        
        #self.adapter = [nn.Sequential(*list(up_conv(feat[i], feat[0], False))).cuda() for i in range(5)]
        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], num_channel)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], num_channel)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], num_channel)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], num_channel)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], num_channel)))
        
        self.fdm_ref = fdm_refine_block(config, num_channel)
        
        self.fdm_exp0 = fdm_expand_block(config, num_channel, [3, 5, 7, 9], k=2)
        self.fdm_exp1 = fdm_expand_block(config, num_channel, [2, 4, 6, 8], k=3)
        self.fdm_exp2 = fdm_expand_block(config, num_channel, [1, 3, 5, 7], k=5)
        self.fdm_exp3 = fdm_expand_block(config, num_channel, [1, 2, 4, 6], k=7)
        self.fdm_exp4 = fdm_expand_block(config, num_channel, [1, 2, 3, 4], k=9)
        
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
        fdm, bcg_map = self.fdm_ref(fdm, xs[2:5])
        
        
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
        OutDict['bcg_map'] = [bcg_map, ]
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
