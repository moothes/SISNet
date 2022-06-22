import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet


NUM = [3, 2, 2, 1, 1]

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
        
def up_conv(cin, cout, k=3):
    yield nn.Conv2d(cin, cout, k, padding=k//2)
    #yield nn.GroupNorm(cout // 2, cout)
    yield nn.GroupNorm(1, cout)
    #yield nn.BatchNorm2d(cout)
    #yield nn.Dropout(0.5)
    yield nn.ReLU(inplace=True)

def initModule(modules):
    for module in modules:
        if type(module) is nn.Conv2d or type(module) is nn.Linear:
            nn.init.kaiming_normal_(module.weight)

def gen_convs(In, Out, num=1):
    for i in range(num):
        yield nn.Conv2d(In, In, 3, padding=1)
        yield nn.BatchNorm2d(In)
        yield nn.ReLU(inplace=True)

def gen_fuse(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.GroupNorm(Out//2, Out)
    #yield nn.BatchNorm2d(Out)
    yield nn.ReLU(inplace=True)

def cp(x, n=2):
    batch, cat, w, h = x.size()
    xn = x.view(batch, cat//n, n, w, h)
    xn = torch.max(xn, dim=2)[0]
    return xn

def gen_final(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.ReLU(inplace=True)

def decode_conv(layer, c):
    for i in range(4 - layer):
        yield nn.Conv2d(c, c, 3, padding=1)
        yield nn.ReLU(inplace=True)
        yield nn.Upsample(scale_factor=2, mode='bilinear')

    yield nn.Conv2d(c, 8, 3, padding=1)
    yield nn.ReLU(inplace=True)

class pred_block(nn.Module):
    def __init__(self, In, Out, up=False):
        super(pred_block, self).__init__()

        self.final_conv = nn.Conv2d(In, Out, 3, padding=1)
        self.pr_conv = nn.Conv2d(Out, 4, 3, padding=1)
        self.up = up

    def forward(self, X):
        a = nn.functional.relu(self.final_conv(X))
        a1 = self.pr_conv(a)
        pred = torch.max(a1, dim=1, keepdim=True)[0]
        if self.up: 
            a = nn.functional.interpolate(a, scale_factor=2, mode='bilinear')
        return [a, pred]

class res_block(nn.Module):
    def __init__(self, cat, layer):
        super(res_block, self).__init__()

        if layer:
            self.conv4 = nn.Sequential(*list(gen_fuse(cat, cat // 2)))

        self.convs = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat//2)))

        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X, encoder):
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
            c = cp(X)
            d = self.conv4(encoder)
            X = torch.cat([c, d], 1)

        X = self.convs(X)
        a = cp(X)
        b = self.conv2(encoder)
        f = torch.cat([a, b], 1)
        f = self.final(f)
        return f

    def initialize(self):
        initModule(self.convs)
        initModule(self.conv2)
        initModule(self.final)

        if self.layer:
            initModule(self.conv4)

class ctr_block(nn.Module):
    def __init__(self, cat, layer):
        super(ctr_block, self).__init__()
        self.conv1 = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat)))
        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X):
        X = self.conv1(X)
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
        X = self.conv2(X)
        x = self.final(X)
        return x

    def initialize(self):
        initModule(self.conv1)
        initModule(self.conv2)
        initModule(self.final)

class final_block(nn.Module):
    def __init__(self, backbone, channel):
        super(final_block, self).__init__()
        self.slc_decode = nn.ModuleList([nn.Sequential(*list(decode_conv(i, channel))) for i in range(5)])
        self.conv = nn.Conv2d(40, 8, 3, padding=1)
        self.backbone = backbone

    def forward(self, xs, phase):
        feats = [self.slc_decode[i](xs[i]) for i in range(5)]
        x = torch.cat(feats, 1)
        
        x = self.conv(x)
        if not self.backbone.startswith('vgg'):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        
        scale = 2 if phase == 'test' else 1
        x = torch.max(x, dim=1, keepdim=True)[0] * scale
        return x

class baseU(nn.Module):
    def __init__(self, feat, config, channel=64):
        super(baseU, self).__init__()
        self.name = 'baseU'
        self.layer = 5

        self.adapters = nn.ModuleList([adapter(in1, channel) for in1 in feat])

        self.slc_blocks = nn.ModuleList([res_block(channel, i) for i in range(self.layer)])
        self.slc_preds = nn.ModuleList([pred_block(channel, channel//2)  for i in range(self.layer)])

        self.ctr_blocks = nn.ModuleList([ctr_block(channel, i) for i in range(self.layer)])
        self.ctr_preds = nn.ModuleList([pred_block(channel, channel//2, up=True)  for i in range(self.layer)])

        self.final = final_block(config['backbone'], channel)
        
        self.cv1 = nn.Sequential(*list(up_conv(128, 64)))
        self.cv2 = nn.Sequential(*list(up_conv(128, 64)))
        self.fdm_block = fdm_refine_block(config, channel)

    def forward(self, encoders, fdm, phase='test'):
        encoders = [self.adapters[i](e_feat) for i, e_feat in enumerate(encoders)]
        
        fdm, bcg_map, base_feat, sup = self.fdm_block(fdm, encoders)
        
        #bf = nn.functional.interpolate(base_feat, size=encoders[-1].size()[-2:], mode='bilinear')
        #first = torch.cat([encoders[-1], bf], dim=1)
        
        slcs, slc_maps = [encoders[-1]], []
        ctrs, ctr_maps = [], []
        stc, cts = None, None

        for i in range(self.layer):
            #print(slcs[-1].shape, encoders[self.layer - 1 - i].shape)
            slc = self.slc_blocks[i](slcs[-1], encoders[self.layer - 1 - i])
            if cts is not None:
                bf = nn.functional.interpolate(base_feat, size=slc.size()[-2:], mode='bilinear')
                slc = torch.cat([cp(slc), cts, bf], dim=1)
            else:
                ctrs.append(slc)
                bf = nn.functional.interpolate(base_feat, size=encoders[-1].size()[-2:], mode='bilinear')
                slc = torch.cat([slc, bf], dim=1)
            slc = self.cv1(slc)
            stc, slc_map = self.slc_preds[i](slc)

            bf = nn.functional.interpolate(base_feat, size=stc.size()[-2:], mode='bilinear')
            ctr = self.ctr_blocks[i](ctrs[-1])
            ctr = torch.cat([cp(ctr), stc, bf], dim=1)
            ctr = self.cv2(ctr)
            cts, ctr_map = self.ctr_preds[i](ctr)

            slcs.append(slc)
            ctrs.append(ctr)
            slc_maps.append(slc_map)
            ctr_maps.append(ctr_map)

        final = self.final(slcs[1:], phase)

        OutPuts = {'final':final, 'sal':slc_maps, 'edge':ctr_maps}
        return OutPuts

class adapter(nn.Module):
    def __init__(self, in1=64, out=64):
        super(adapter, self).__init__()
        self.reduce = in1 // 64
        self.conv = nn.Conv2d(out, out, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        batch, cat, height, width = X.size()
        X = torch.max(X.view(batch, 64, self.reduce, height, width), dim=2)[0]
        X = self.relu(self.conv(X))

        return X


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

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        c = 64
        self.encoder = encoder
        self.decoder = baseU(feat, config, c)
    
    def forward(self, x, fdm, phase='test'):
        enc_feats = self.encoder(x)
        
        
        
        OutDict = self.decoder(enc_feats, fdm, phase='test')
        return OutDict
