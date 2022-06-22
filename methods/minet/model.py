import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fuse = nn.Sequential(*list(up_conv(192, feat)))
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

def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


class SIM(nn.Module):
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(h_C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h


class conv_2nV1(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(conv_2nV1, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc, out_c, 1)

        else:
            raise NotImplementedError

    def forward(self, in_h, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        if self.main == 0:
            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            # stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
            # 这里使用的不是in_h，而是h
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            # stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

        return out


class conv_3nV1(nn.Module):
    def __init__(self, in_hc=64, in_mc=256, in_lc=512, out_c=64):
        super(conv_3nV1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.AvgPool2d((2, 2), stride=2)

        mid_c = min(in_hc, in_mc, in_lc)
        self.relu = nn.ReLU(True)

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        # stage 2
        self.h2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        # stage 3
        self.m2m_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc, out_c, 1)

    def forward(self, in_h, in_m, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(self.upsample(m))

        h2m = self.h2m_1(self.downsample(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))

        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        # stage 2
        h2m = self.h2m_2(self.downsample(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        # stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        return out


class AIM(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(AIM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)

    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
        out_xs.append(self.conv3(xs[2], xs[3], xs[4]))
        out_xs.append(self.conv4(xs[3], xs[4]))

        return out_xs

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.encoder = encoder
        self.config = config
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.inter_feat = (64, 64, 64, 64, 64)
        if config['backbone'] == 'resnet':
            self.upconv0 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.trans = AIM(iC_list=feat, oC_list=self.inter_feat)

        self.sim32 = SIM(self.inter_feat[4], self.inter_feat[4] // 2)
        self.sim16 = SIM(self.inter_feat[3], self.inter_feat[3] // 2)
        self.sim8 = SIM(self.inter_feat[2], self.inter_feat[2] // 2)
        self.sim4 = SIM(self.inter_feat[1], self.inter_feat[1] // 2)
        self.sim2 = SIM(self.inter_feat[0], self.inter_feat[0] // 2)

        self.upconv5 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)
        
        self.fdm_block = fdm_refine_block(config, 64)

    def forward(self, x, fdm, phase='test'):
        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = self.encoder(x)
        
        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = self.trans(
            in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        )
        
        fdm, bcg_map, base_feat, sup = self.fdm_block(fdm, [in_data_2, in_data_4, in_data_8, in_data_16, in_data_32])

        bf = F.interpolate(base_feat, size=in_data_32.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        out_data_32 = self.upconv5(self.sim32(in_data_32) + bf)  # 1024

        out_data_16 = self.upsample_add(out_data_32, in_data_16)  # 1024
        bf = F.interpolate(base_feat, size=out_data_16.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        out_data_16 = self.upconv4(self.sim16(out_data_16) + bf)

        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        bf = F.interpolate(base_feat, size=out_data_8.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        out_data_8 = self.upconv3(self.sim8(out_data_8) + bf)  # 512

        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        bf = F.interpolate(base_feat, size=out_data_4.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        out_data_4 = self.upconv2(self.sim4(out_data_4) + bf)  # 256

        #print(out_data_4.size(), in_data_2.size())
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        bf = F.interpolate(base_feat, size=out_data_2.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        out_data_2 = self.upconv1(self.sim2(out_data_2) + bf)  # 64

        if self.config['backbone'] == 'resnet':
            out_data_2 = self.upconv0(self.upsample(out_data_2, scale_factor=2))  # 32
        out_data = self.classifier(out_data_2)

        out_dict = {}
        out_dict['sal'] = [out_data]
        out_dict['final'] = out_data
        return out_dict
