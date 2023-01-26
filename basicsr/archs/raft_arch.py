import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from basicsr.archs.raft.core.update import BasicUpdateBlock, SmallUpdateBlock
from basicsr.archs.raft.core.extractor import BasicEncoder, SmallEncoder
from basicsr.archs.raft.core.corr import CorrBlock, AlternateCorrBlock
from basicsr.archs.raft.core.utils.utils import bilinear_sampler, coords_grid, upflow8
from basicsr.utils.registry import ARCH_REGISTRY

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

@ARCH_REGISTRY.register()
class RAFT(nn.Module):
    def __init__(self, use_small=True, raft_path=None):
        super(RAFT, self).__init__()

        if use_small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4


        self.dropout = 0
        self.alternate_corr = False

        # feature network, context network, and update block
        if use_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
        
        if raft_path:
            load_raft = torch.load(raft_path, map_location=lambda storage, loc: storage)
            state_dict = OrderedDict()
            for k, v in load_raft.items():
                k = k.replace('module.', '')
                state_dict[k] = v
            self.load_state_dict(state_dict)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=False):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=False):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=False):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            if itr == iters - 1:
                return flow_up


