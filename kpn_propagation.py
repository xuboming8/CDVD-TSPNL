import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.archs.kpn import KernelConv
from basicsr.archs.kpn_pixel import IDynamicDWConv
from basicsr.archs.kpn_channel import DynamicDWConv


class propagation_backward(nn.Module):
    def __init__(self, num_feat=64, kernel=3):
        super().__init__()
        self.num_feat = num_feat
        # self.kernel_conv = IDynamicDWConv(channels=num_feat, kernel_size=kernel, group_channels=num_feat)
        self.kernel_conv = DynamicDWConv(dim=num_feat, kernel_size=kernel, groups=num_feat)
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)

        self.project_in1 = nn.Conv2d(num_feat, num_feat * 4, kernel_size=1)
        self.dwconv1 = nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, groups=num_feat * 4)
        self.project_out1 = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1)
        self.project_in2 = nn.Conv2d(num_feat, num_feat * 4, kernel_size=1)
        self.dwconv2 = nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, groups=num_feat * 4)
        self.project_out2 = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []
        feat_prop = feature[:, t - 1, :, :, :]
        backward_list.append(feat_prop)
        # propagation
        for i in range(t - 2, -1, -1):
            x_feat = feature[:, i, :, :, :]
            feat_prop = self.kernel_conv(x_feat, feat_prop)
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))   # b 128 256 256
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.project_out1(self.dwconv1(self.project_in1(feat_prop1))))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.project_out2(self.dwconv2(self.project_in2(feat_prop2))))
            feat_prop = feat_prop1 + feat_prop2
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=1)      # b t 64 256 256
        return conv3d_feature


class propagation_forward(nn.Module):
    def __init__(self, num_feat=64, kernel=3):
        super().__init__()
        self.num_feat = num_feat
        # self.kernel_conv = IDynamicDWConv(channels=num_feat, kernel_size=kernel, group_channels=num_feat)
        self.kernel_conv = DynamicDWConv(dim=num_feat, kernel_size=kernel, groups=num_feat)
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)

        self.project_in1 = nn.Conv2d(num_feat, num_feat * 4, kernel_size=1)
        self.dwconv1 = nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, groups=num_feat * 4)
        self.project_out1 = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1)
        self.project_in2 = nn.Conv2d(num_feat, num_feat * 4, kernel_size=1)
        self.dwconv2 = nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, groups=num_feat * 4)
        self.project_out2 = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                          # b t 64 256 256
        forward_list = []
        feat_prop = feature[:, 0, :, :, :]
        forward_list.append(feat_prop)
        for i in range(1, t):
            x_feat = feature[:, i, :, :, :]
            feat_prop = self.kernel_conv(x_feat, feat_prop)
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.project_out1(self.dwconv1(self.project_in1(feat_prop1))))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.project_out2(self.dwconv2(self.project_in2(feat_prop2))))
            feat_prop = feat_prop1 + feat_prop2
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=1)      # b t 64 256 256
        return conv3d_feature


class propagation(nn.Module):
    def __init__(self, num_feat=64, kernel=3):
        super().__init__()
        self.pro_for = propagation_forward(num_feat, kernel)
        self.pro_back = propagation_backward(num_feat, kernel)

    def forward(self, feature):
        feature = self.pro_for(self.pro_back(feature))
        return feature


# net = propagation(num_feat=64, kernel=3).cuda()
# x = torch.randn(2, 10, 64, 256, 256).cuda()
# y = net(x)
# print(y.shape)

