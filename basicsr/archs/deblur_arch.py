import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer, ResidualBlockNoBN2D
from basicsr.archs.global_attention import globalAttention
from basicsr.archs.srn import SRN
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.spynet_arch import SpyNet
import numpy as np


@ARCH_REGISTRY.register()
class Deblur(nn.Module):
    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # transformer
        self.global_attention = globalAttention(num_feat)
        self.conv_downsample = nn.Conv3d(num_feat, num_feat, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        self.conv_upsample = nn.ConvTranspose3d(num_feat, num_feat, (3, 4, 4), (1, 2, 2), (1, 1, 1))

        # extractor
        self.extractor_2dres = extractor_2DRes(3, num_feat, 3)

        # propogation branch
        self.forward_propagation = manual_conv3d_propagation_forward(num_feat, num_block)
        self.backward_propagation = manual_conv3d_propagation_backward(num_feat, num_block)
        self.fusion = nn.Conv3d(num_feat * 2 + 1, num_feat, 1, 1, 0, bias=True)

        # reconstruction
        self.srn = SRN()

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, t, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, t - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def calc_meanFilter(self, img, kernel_size=11):
        mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        mean_filter_X = torch.from_numpy(mean_filter_X).float().type_as(img)
        new_img = torch.zeros_like(img)
        for i in range(img.size()[1]):
            new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                                 stride=1, padding=kernel_size // 2)
        return new_img

    def get_TSP(self, ref_img, nbr_img_list):
        with torch.no_grad():
            ref_img = self.calc_meanFilter(ref_img, kernel_size=5)
            nbr_img_list = [self.calc_meanFilter(im, kernel_size=5) for im in nbr_img_list]
            delta = 1.
            diff = torch.zeros_like(ref_img)
            for nbr_img in nbr_img_list:
                diff = diff + (nbr_img - ref_img).pow(2)
            diff = diff / (2 * delta * delta)
            diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
            tsp = torch.exp(-diff)
            return tsp

    def one_stage(self, lrs):
        flows_forward, flows_backward = self.get_flow(lrs)
        b, t, c, h, w = lrs.size()

        # extractor
        lrs_feature = self.extractor_2dres(lrs)

        # transformer
        tf_intput_feature = self.conv_downsample(lrs_feature)   # b c t h//2 w//2
        tf_output_feature = rearrange(self.global_attention(rearrange(tf_intput_feature, 'b c t h w -> b t c h w')), 'b t c h w -> b c t h w')
        tf_output_feature = self.conv_upsample(tf_output_feature)  # b c t h w
        tf_output_feature = rearrange(lrs_feature + tf_output_feature, 'b c t h w -> b t c h w')

        # backward & forward propagation & res block
        backward_feature = self.backward_propagation(tf_output_feature, flows_backward)  # b 64 t 256 256
        forward_feature = self.forward_propagation(tf_output_feature, flows_forward)  # b 64 t 256 256

        # compute TSP
        tsp_l = []
        with torch.no_grad():
            for i in range(t):
                this_f = tf_output_feature[:, i, :, :, :]
                if i == 0:
                    right_f = flow_warp(backward_feature[:, :, i + 1, :, :], rearrange(flows_backward[:, i, :, :, :], 'b c h w -> b h w c'))
                    tsp = self.get_TSP(this_f, [this_f, right_f])
                elif i == t - 1:
                    left_f = flow_warp(forward_feature[:, :, i - 1, :, :], rearrange(flows_forward[:, i - 1, :, :, :], 'b c h w -> b h w c'))
                    tsp = self.get_TSP(this_f, [left_f, this_f])
                else:
                    left_f = flow_warp(forward_feature[:, :, i - 1, :, :], rearrange(flows_forward[:, i - 1, :, :, :], 'b c h w -> b h w c'))
                    right_f = flow_warp(backward_feature[:, :, i + 1, :, :], rearrange(flows_backward[:, i, :, :, :], 'b c h w -> b h w c'))
                    tsp = self.get_TSP(this_f, [left_f, right_f])
                tsp_l.append(tsp)
        tsp = torch.stack(tsp_l, dim=2)

        # fusion backward&forward feature
        fusion_cat = torch.cat([backward_feature, forward_feature, tsp], dim=1)  # b 128 10 256 256
        fusion_cat = self.lrelu(self.fusion(fusion_cat))  # b 64 t 256 256

        # reconstruction
        out = rearrange(self.srn(rearrange(fusion_cat, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
        return out.contiguous() + lrs

    def forward(self, lrs):
        # time_start = time.time()
        out = self.one_stage(self.one_stage(lrs))
        # time_end = time.time()
        # print("inference time:", time_end-time_start)
        return out


class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=30):
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        return self.main(fea)


class extractor_2DRes(nn.Module):
    def __init__(self, inchan=3, num_feat=64, num_block=30):
        super().__init__()
        self.conv = nn.Conv3d(inchan, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN2D, num_block, num_feat=num_feat))

    def forward(self, feat):
        out = self.main(self.conv(rearrange(feat, 'b t c h w -> b c t h w')))
        return out


class manual_conv3d_propagation_backward(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.conv_downchan = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature, flows_backward):
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        for i in range(t - 1, -1, -1):
            x_feat = feature[:, i, :, :, :]
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, rearrange(flow, 'b c h w -> b h w c'))
            # fusion propagation
            x_feat = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256
            feat_prop = self.lrelu(self.conv_downchan(x_feat))   # b 64 256 256
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=2)      # b 64 t 256 256
        return conv3d_feature


class manual_conv3d_propagation_forward(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.conv_downchan = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature, flows_forward):
        b, t, c, h, w = feature.size()                          # b t 64 256 256
        forward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        for i in range(0, t):
            x_feat = feature[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, rearrange(flow, 'b c h w -> b h w c'))
            # fusion propagation
            x_feat = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256
            feat_prop = self.lrelu(self.conv_downchan(x_feat))   # b 64 256 256
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=2)      # b 64 t 256 256
        return conv3d_feature
