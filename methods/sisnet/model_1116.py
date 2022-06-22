import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout, k=3):
    yield nn.Conv2d(cin, cout, k, padding=k//2)
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

    def forward(self, x0, sc, fore_map):
        x0 = nn.functional.interpolate(x0, size=sc.size()[2:], mode='bilinear')
        fore_map = nn.functional.interpolate(fore_map, size=sc.size()[2:], mode='bilinear')
        
        #res_x0 = torch.sigmoid(self.res_conv1(x0 - sc))
        #sup = self.sup(x0)
        #res_x1 = torch.sigmoid(sup)
        #x0 = self.fuse(torch.cat([x0 * res_x0, sc, sc * res_x1], dim=1))
        res_x0 = torch.sigmoid(self.res_conv1(x0 - sc))
        sup = self.sup(x0)
        res_x1 = torch.sigmoid(sup)
        x0 = self.fuse(torch.cat([x0, sc * res_x0, x0 * res_x1], dim=1))
        
        return x0, sup
        
def aspp_channel(fdm, block, k, s):
    fdm1 = F.max_pool2d(fdm, kernel_size=k, stride=s, padding=k//2)
    fdm1 = block(fdm1)
    fdm1 = F.interpolate(fdm1, size=fdm.shape[2:], mode='bilinear')
    return fdm1

class ASPP_block(nn.Module):
    def __init__(self, config, feat):
        super(ASPP_block, self).__init__()
        aspp_feat = feat // 4
        self.conv0 = nn.Sequential(*list(up_conv(1, aspp_feat)))
        self.conv1 = nn.Sequential(*list(up_conv(aspp_feat, aspp_feat)))
        self.conv2 = nn.Sequential(*list(up_conv(aspp_feat, aspp_feat)))
        self.conv3 = nn.Sequential(*list(up_conv(aspp_feat, aspp_feat)))
        self.conv4 = nn.Sequential(*list(up_conv(aspp_feat, aspp_feat)))
        self.fuse_conv1 = nn.Sequential(*list(up_conv(feat, feat)))

    def forward(self, fdm):
        fdm = self.conv0(fdm)
        
        fdm1 = aspp_channel(fdm, self.conv1, 13, 12)
        fdm2 = aspp_channel(fdm, self.conv2, 9, 8)
        fdm3 = aspp_channel(fdm, self.conv3, 7, 5)
        fdm4 = aspp_channel(fdm, self.conv4, 5, 2)
        
        fdm_exp = torch.cat([fdm1, fdm2, fdm3, fdm4], dim=1)
        fdm_exp = self.fuse_conv1(fdm_exp)
        
        return fdm_exp
        
class fdm_refine_block(nn.Module):
    def __init__(self, config, feat, ds=[3, 5, 7, 9]):
        super(fdm_refine_block, self).__init__()
        self.fuse = nn.Sequential(*list(up_conv(feat * 3, feat)))
        self.bcg_conv = nn.Conv2d(feat, 1, 1)
        self.aspp_block = ASPP_block(config, feat)
        self.sup = nn.Conv2d(feat, 1, 1)
        
        self.fuse0 = nn.Sequential(*list(up_conv(feat, feat)))
        self.fuse1 = nn.Sequential(*list(up_conv(1, feat)))
        
        self.k = 7
        self.nl_size = 30

    def forward(self, fdm, xs):
        x0 = F.interpolate(xs[0], size=fdm.shape[2:], mode='bilinear')
        x1 = F.interpolate(xs[1], size=fdm.shape[2:], mode='bilinear')
        x2 = F.interpolate(xs[2], size=fdm.shape[2:], mode='bilinear')
        x3 = F.interpolate(xs[3], size=fdm.shape[2:], mode='bilinear')
        x4 = F.interpolate(xs[4], size=fdm.shape[2:], mode='bilinear')
        
        base_feat = self.fuse(torch.cat([x2, x3, x4], dim=1))
        bcg_map = torch.sigmoid(self.bcg_conv(base_feat))
        
        mean_bcg = torch.mean(bcg_map, dim=(2,3), keepdim=True)
        fdm = fdm * (bcg_map > mean_bcg).float()
        
        nl_feat = self.fuse0(base_feat)
        nl_feat = F.interpolate(nl_feat, size=(self.nl_size, self.nl_size), mode='bilinear')
        #nl_fdm = self.fuse1(fdm)
        #nl_fdm = F.interpolate(fdm, size=(self.nl_size, self.nl_size), mode='bilinear') * nl_feat
        nl_fdm = F.adaptive_max_pool2d(fdm, (self.nl_size, self.nl_size)) * nl_feat
        b, c, _, _ = nl_fdm.size()
        nl_0 = nl_fdm.view(b, c, -1, 1)
        nl_1 = nl_feat.view(b, c, 1, -1)
        #nl = torch.max(F.cosine_similarity(nl_0, nl_1, dim=1), dim=-2)[0] + 1
        nl = torch.max(torch.sum(nl_0 * nl_1, dim=1), dim=2)[0]
        fdm_exp = nl.view(b, 1, self.nl_size, self.nl_size)
        fdm_exp = F.interpolate(fdm_exp, size=fdm.shape[2:], mode='bilinear')
        #print(nl_fdm.shape, nl_feat.shape, nl.shape)
        #fdm = F.max_pool2d(fdm, kernel_size=self.k, stride=1, padding=(self.k - 1) // 2)
        #fdm = F.avg_pool2d(fdm, kernel_size=self.k, stride=1, padding=(self.k - 1) // 2)
        #fdm_exp = self.aspp_block(fdm)
        base_feat = fdm_exp * base_feat
        sup = self.sup(base_feat)
        
        return fdm, bcg_map, base_feat, sup

class fdm_fuse_block(nn.Module):
    def __init__(self, config, feat):
        super(fdm_fuse_block, self).__init__()
        self.conv0 = nn.Sequential(*list(up_conv(feat, feat)))
        self.fuse = nn.Sequential(*list(up_conv(feat * 2, feat)))

    def forward(self, atten, x):
        att = F.interpolate(atten, size=x.shape[2:], mode='bilinear')
        att = self.conv0(att)
        x0 = self.fuse(torch.cat([att * x, x], dim=1))
        return x0

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
        
        self.fdm_exp0 = fdm_fuse_block(config, num_channel)
        self.fdm_exp1 = fdm_fuse_block(config, num_channel)
        self.fdm_exp2 = fdm_fuse_block(config, num_channel)
        self.fdm_exp3 = fdm_fuse_block(config, num_channel)
        self.fdm_exp4 = fdm_fuse_block(config, num_channel)
        
        self.info0 = info_block(config, num_channel)
        self.info1 = info_block(config, num_channel)
        self.info2 = info_block(config, num_channel)
        self.info3 = info_block(config, num_channel)
        self.info4 = info_block(config, num_channel)
        
        self.pfuse = nn.Sequential(*list(up_conv(num_channel * 5, num_channel)))
        self.final_pred = nn.Conv2d(num_channel, 1, 1)
        
        
    def forward(self, xs, fdm, x_size):
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])
        
        fdm = F.adaptive_max_pool2d(fdm, xs[0].size()[2:])
        fdm, bcg_map, feat, sup0 = self.fdm_ref(fdm, xs[0:5])
        
        x4 = self.fdm_exp0(feat, xs[4])
        x3 = self.fdm_exp1(feat, xs[3])
        x2 = self.fdm_exp2(feat, xs[2])
        x1 = self.fdm_exp3(feat, xs[1])
        x0 = self.fdm_exp4(feat, xs[0])
        
        p4, sup_4 = self.info0(x4, x4, fdm)
        p3, sup_3 = self.info1(p4, x3, fdm)
        p2, sup_2 = self.info2(p3, x2, fdm)
        p1, sup_1 = self.info3(p2, x1, fdm)
        p0, sup_0 = self.info4(p1, x0, fdm)
        
        p4 = F.interpolate(p4, size=p0.shape[2:], mode='bilinear')
        p3 = F.interpolate(p3, size=p0.shape[2:], mode='bilinear')
        p2 = F.interpolate(p2, size=p0.shape[2:], mode='bilinear')
        p1 = F.interpolate(p1, size=p0.shape[2:], mode='bilinear')
        
        feat = torch.cat([p4, p3, p2, p1, p0], dim=1)
        p = self.pfuse(feat)
        pred = self.final_pred(p)
        #print(pred.shape)

        #print(torch.max(sup0))
        OutDict = {}
        #OutDict['atten'] = [r1, r2, r3, r4]
        OutDict['fdm'] = fdm
        OutDict['bcg_map'] = [bcg_map, ]
        #OutDict['ctr'] = [bup_0, bup_1, bup_2, bup_3, bup_4]
        OutDict['sal'] = [pred, sup0, sup_0, sup_1, sup_2, sup_3, sup_4]
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
