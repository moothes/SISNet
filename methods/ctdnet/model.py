import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

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

class fdm_refine_block(nn.Module):
    def __init__(self, config, feat, ds=[3, 5, 7, 9]):
        super(fdm_refine_block, self).__init__()
        self.fuse = nn.Sequential(*list(up_conv(3584, feat)))
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        # left = F.relu(left_1 * right_1, inplace=True)
        # right = F.relu(left_2 * right_2, inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

        
class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        block = BasicBlock
        self.encoder = encoder

        self.path1_1 = nn.Sequential(
            conv1x1(feat[4], 64),
            nn.BatchNorm2d(64)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(feat[4], 64),
            nn.BatchNorm2d(64)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(feat[3], 64),
            nn.BatchNorm2d(64)
        )

        self.path2 = SAM(feat[2], 64)

        self.path3 = nn.Sequential(
            conv1x1(feat[1], 64),
            nn.BatchNorm2d(64)
        )

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse12 = CAM(64)
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)
        
        self.fdm_block = fdm_refine_block(config, 64)
        self.cv1 = conv3x3(128, 64, bias=True)


    def forward(self, x, fdm, phase='test'):
        shape = x.size()[2:]
        l1, l2, l3, l4, l5 = self.encoder(x)
        
        fdm = F.interpolate(fdm, size=l3.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        fdm, bcg_map, base_feat, sup = self.fdm_block(fdm, [l1, l2, l3, l4, l5])

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        path1_1 = self.path1_1(path1_1)
        bf1_1 = F.interpolate(base_feat, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        path1_1 = self.cv1(torch.cat([bf1_1,path1_1], dim=1))
        path1_2 = F.relu(self.path1_2(l5), inplace=True)                                            # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)                                                    # 1/32
        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)   # 1/16

        path1_3 = F.relu(self.path1_3(l4), inplace=True)                                            # 1/16
        path1 = self.fuse1_2(path1_2, path1_3)                                                      # 1/16
        # path1 = F.interpolate(path1, size=l3.size()[2:], mode='bilinear', align_corners=True)

        path2 = self.path2(l3)                                                                      # 1/8
        path12 = self.fuse12(path1, path2)                                                          # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = F.relu(self.path3(l2), inplace=True)                                              # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)   # 1/4
        path3 = self.fuse3(path3_1, path3_2)                                                        # 1/4

        path_out = self.fuse23(path12, path3)                                                       # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)

        logits_2 = F.interpolate(self.head_2(path12), size=shape, mode='bilinear', align_corners=True)
        logits_3 = F.interpolate(self.head_3(path1), size=shape, mode='bilinear', align_corners=True)
        logits_4 = F.interpolate(self.head_4(path1_2), size=shape, mode='bilinear', align_corners=True)
        logits_5 = F.interpolate(self.head_5(path1_1), size=shape, mode='bilinear', align_corners=True)
        
        OutDict = {}
        OutDict['final'] = logits_1
        OutDict['edge'] = [logits_edge, ]
        OutDict['sal'] = [logits_1, logits_2, logits_3, logits_4, logits_5]
        
        
        return OutDict
