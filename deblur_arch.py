import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from basicsr.archs.wave_tf import DWT, IWT
from basicsr.archs.kpn_propagation import propagation


@ARCH_REGISTRY.register()
class Deblur(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat

        # extractor & reconstruction
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)

        # wave tf
        self.dwt = DWT()
        self.iwt = IWT()
        self.x_wave_1_12_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_1_12_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_1_21_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_1_21_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_1_22_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_1_22_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        # wave pro
        self.x_wave_2_12_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_2_12_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_2_21_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_2_21_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_2_22_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x_wave_2_22_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)

        self.pro1 = propagation(num_feat=num_feat, kernel=3)
        self.pro2 = propagation(num_feat=num_feat, kernel=3)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lrs):
        B, T, _, _, _ = lrs.size()
        # reflection
        if(B == 1):  # test
            test_frame = 10
            len = (test_frame - T % test_frame) % test_frame
            for i in range(0, len):
                lrs = torch.cat([lrs, lrs[:, T - 1 - i, :, :, :].unsqueeze(1)], dim=1)

        # time_start = time.time()
        b, t, c, h, w = lrs.size()
        lrs_feature = self.feat_extractor(lrs.permute(0, 2, 1, 3, 4))  # b c t h w

        # transformer VSR
        tf_list = []
        for i in range(0, b):
            x_batch = lrs_feature[i, :, :, :, :].permute(1, 0, 2, 3)
            # scale1
            tf_wave1 = self.dwt(x_batch).contiguous().view(4, t, self.num_feat, h // 2, w // 2)
            x1_11 = tf_wave1[0, :, :, :, :]
            x1_12 = self.x_wave_1_12_conv2(self.lrelu(self.x_wave_1_12_conv1(tf_wave1[1, :, :, :, :]))).unsqueeze(dim=0)
            x1_21 = self.x_wave_1_21_conv2(self.lrelu(self.x_wave_1_21_conv1(tf_wave1[2, :, :, :, :]))).unsqueeze(dim=0)
            x1_22 = self.x_wave_1_22_conv2(self.lrelu(self.x_wave_1_22_conv1(tf_wave1[3, :, :, :, :]))).unsqueeze(dim=0)
            # scale2
            tf_wave2 = self.dwt(x1_11).contiguous().view(4, t, self.num_feat, h // 4, w // 4)
            x2_11 = self.pro2(tf_wave2[0, :, :, :, :].unsqueeze(dim=0))
            x2_12 = self.x_wave_2_12_conv2(self.lrelu(self.x_wave_2_12_conv1(tf_wave2[1, :, :, :, :]))).unsqueeze(dim=0)
            x2_21 = self.x_wave_2_21_conv2(self.lrelu(self.x_wave_2_21_conv1(tf_wave2[2, :, :, :, :]))).unsqueeze(dim=0)
            x2_22 = self.x_wave_2_22_conv2(self.lrelu(self.x_wave_2_22_conv1(tf_wave2[3, :, :, :, :]))).unsqueeze(dim=0)
            tf_wave2 = torch.cat([x2_11, x2_12, x2_21, x2_22], dim=0)
            x1_11 = self.pro1(self.iwt(tf_wave2.contiguous().view(4 * t, self.num_feat, h // 4, w // 4)).unsqueeze(dim=0))
            tf_wave1 = torch.cat([x1_11, x1_12, x1_21, x1_22], dim=0)
            tf_wave1 = self.iwt(tf_wave1.contiguous().view(4 * t, self.num_feat, h // 2, w // 2))
            tf_list.append(tf_wave1.unsqueeze(dim=0))
        tf_output_feature = torch.cat(tf_list, dim=0)

        # reconstruction
        out = self.recons(tf_output_feature.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        out = out.contiguous() + lrs

        # time_end = time.time()
        # print("inference time:", time_end - time_start)
        return out[:, 0:T, :, :, :]