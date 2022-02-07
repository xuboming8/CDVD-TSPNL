import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger
from basicsr.archs.layers.create_act import get_act_layer
from basicsr.archs.cupy_layers.aggregation_zeropad import LocalConvolution
from basicsr.archs.tsm import TemporalShift
import datetime

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN3D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN3D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlockNoBN2D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN2D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class TSM(nn.Module):
    def __init__(self, num_feat=64):
        super(TSM, self).__init__()
        self.tsm = TemporalShift(nn.Sequential(), n_div=8, inplace=False)
        self.conv = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # split segment & tsm3
        if(x.size(1) == 10):
            x_list = torch.split(x, [3, 3, 4], dim=1)
            x0 = self.tsm(x_list[0])
            x1 = self.tsm(x_list[1])
            x2 = self.tsm(x_list[2])
            x = torch.cat([x0, x1, x2], dim=1)  # b t c h w
        else:
            x_list = torch.split(x, 4, dim=1)
            if(len(x_list)==1):
                x = self.tsm(x_list[0])
            else:
                x0 = self.tsm(x_list[0])
                x1 = self.tsm(x_list[1])
                x = torch.cat([x0, x1], dim=1)  # b t c h w
        # tsm residual block
        x = x.permute(0, 2, 1, 3, 4)            # b c t h w
        identity = x
        out = self.relu(self.conv(x) + identity)
        return out.permute(0, 2, 1, 3, 4)       # b t c h w


def manual_padding_1(lrs):
    x_0 = lrs[:, 1, :, :, :].unsqueeze(1)
    x_t = lrs[:, -2, :, :, :].unsqueeze(1)
    lrs = torch.cat([x_0, lrs], dim=1)
    lrs = torch.cat([lrs, x_t], dim=1)
    return lrs

class conv2d_extractor(nn.Module):
    def __init__(self, num_feat=64):
        super(conv2d_extractor, self).__init__()
        self.conv3d_extractor = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)

    def forward(self, x):
        lrs_feature = self.conv3d_extractor(x)  # b 64 t 256 256
        return lrs_feature

###############################
# RCAB
###############################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlock2D_fft(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock2D_fft, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1),
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=1, stride=1),
        )
        self.norm = 'backward'

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class ResidualBlock3D_fft(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock3D_fft, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, 3, 1, 1),
        )
        self.main_fft = nn.Sequential(
            nn.Conv3d(num_feat * 2, num_feat * 2, (3, 1, 1), 1, (1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat * 2, num_feat * 2, (3, 1, 1), 1, (1, 0, 0)),
        )
        self.norm = 'backward'

    def forward(self, x):
        B, T, C, H, W = x.shape
        fft_list = []
        for i in range(0, T):
            feat_now = x[:, i, :, :, :]
            y = torch.fft.rfft2(feat_now, norm=self.norm)
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            fft_list.append(y_f)
        fft_y = torch.stack(fft_list, dim=2)
        fft_y = self.main_fft(fft_y)

        rfft_list = []
        for i in range(0, T):
            feat_now = fft_y[:, :, i, :, :]
            y_real, y_imag = torch.chunk(feat_now, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            rfft_list.append(y)
        rfft_y = torch.stack(rfft_list, dim=2)
        return self.main(x.permute(0, 2, 1, 3, 4)) + x.permute(0, 2, 1, 3, 4) + rfft_y


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    b, c, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # mask
    # mask = torch.autograd.Variable(torch.ones(b, 1, h, w)).cuda()
    # mask = F.grid_sample(mask, vgrid)
    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)



## CotNet
class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, 1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2 * dim, dim // factor, 3, 1, 1, bias=False),
            nn.BatchNorm3d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // factor, pow(kernel_size, 2) * dim // share_planes, 3, 1, 1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm3d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 3, 1, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix * dim, 3, 1, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)           # 4 64 10 64 64
        qk = torch.cat([x, k], dim=1)
        b, c, t, qk_hh, qk_ww = qk.size()  # 4 128 10 64 64

        w = self.embed(qk)              # 4 (3*3*8) 10 64 64
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, t, qk_hh, qk_ww)    # 4 1 8 9 10 64 64

        x = self.conv1x1(x)             # 4 64 10 64 64
        list = []
        for i in range(0, t):
            tmp = self.local_conv(x[:,:,i,:,:],w[:,:,:,:,i,:,:])
            list.append(tmp)

        x = torch.stack(list, dim=2)    # 4 64 10 64 64
        x = self.bn(x)
        x = self.act(x)

        B, C, T, H, W = x.shape
        x = x.view(B, C, 1, T, H, W)       # 4 64 1 10 256 256
        k = k.view(B, C, 1, T, H, W)       # 4 64 1 10 256 256
        x = torch.cat([x, k], dim=2)       # 4 64 2 10 256 256

        x_gap = x.sum(dim=2)            # 4 64 10 256 256
        x_gap = x_gap.mean((3, 4), keepdim=True)    # 4 64 10 1 1
        x_attn = self.se(x_gap)         # 4 128 10 1 1
        x_attn = x_attn.view(B, C, self.radix, T)    # 4 64 2 10
        x_attn = F.softmax(x_attn, dim=2)         # 4 64 2 10
        out = (x * x_attn.reshape((B, C, self.radix, T, 1, 1))).sum(dim=2)

        return out.contiguous()


class CotLayer_(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer_, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, 1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2 * dim, dim // factor, (3, 1, 1), 1, (1, 0, 0), bias=False),
            nn.BatchNorm3d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // factor, pow(kernel_size, 2) * dim // share_planes, (3, 1, 1), 1, (1, 0, 0)),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, (3, 1, 1), 1, (1, 0, 0), dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm3d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, (3, 1, 1), 1, (1, 0, 0)),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix * dim, (3, 1, 1), 1, (1, 0, 0))
        )

    def forward(self, x):
        k = self.key_embed(x)           # 4 64 10 64 64
        qk = torch.cat([x, k], dim=1)
        b, c, t, qk_hh, qk_ww = qk.size()  # 4 128 10 64 64

        w = self.embed(qk)              # 4 (3*3*8) 10 64 64
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, t, qk_hh, qk_ww)    # 4 1 8 9 10 64 64

        x = self.conv1x1(x)             # 4 64 10 64 64
        list = []
        for i in range(0, t):
            tmp = self.local_conv(x[:,:,i,:,:],w[:,:,:,:,i,:,:])
            list.append(tmp)

        x = torch.stack(list, dim=2)    # 4 64 10 64 64
        x = self.bn(x)
        x = self.act(x)

        B, C, T, H, W = x.shape
        x = x.view(B, C, 1, T, H, W)       # 4 64 1 10 256 256
        k = k.view(B, C, 1, T, H, W)       # 4 64 1 10 256 256
        x = torch.cat([x, k], dim=2)       # 4 64 2 10 256 256

        x_gap = x.sum(dim=2)            # 4 64 10 256 256
        x_gap = x_gap.mean((3, 4), keepdim=True)    # 4 64 10 1 1
        x_attn = self.se(x_gap)         # 4 128 10 1 1
        x_attn = x_attn.view(B, C, self.radix, T)    # 4 64 2 10
        x_attn = F.softmax(x_attn, dim=2)         # 4 64 2 10
        out = (x * x_attn.reshape((B, C, self.radix, T, 1, 1))).sum(dim=2)

        return out.contiguous()


class CotLayer2d(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer2d, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            # nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            # nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            # nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, base_width=64, act_layer=nn.ReLU):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)))
        first_planes = width
        outplanes = planes

        self.conv1 = nn.Conv3d(inplanes, first_planes, kernel_size=1, bias=False)
        self.act1 = act_layer(inplace=True)
        self.conv2 = CotLayer_(width, kernel_size=3)
        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False)
        self.act3 = act_layer(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += residual
        x = self.act3(x)

        return x


# model = CotLayer(dim=64, kernel_size=3).cuda()
# input = torch.randn(4, 64, 10, 64, 64).cuda()
# b, t, c, h, w = input.shape
# input = input.contiguous().view(-1, c, h, w)
# out = model(input)
# print(out.shape)

# device = torch.device('cuda:0')
# # model = Bottleneck(inplanes=64, planes=64).to(device)
# model = CotLayer(dim=64, kernel_size=3).to(device)
# input = torch.randn(4, 64, 10, 128, 128).to(device)
# start = datetime.datetime.now()
# for i in range(1, 100):
#     out = model(input)
#     print(i, out.shape)
# end = datetime.datetime.now()
# print(end - start)

# fft = ResidualBlock3D_fft(num_feat=64)
# x = torch.randn(2,10,64,256,256)
# y = fft(x)
# print(y.shape)
