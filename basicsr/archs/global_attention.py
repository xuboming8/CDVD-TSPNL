import torch
from torch import nn as nn
import cv2
from torch.nn import functional as F
import datetime


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, stride_q=8, stride_kv=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.num_feat = num_feat
        self.patch_size = patch_size
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, groups=num_feat)
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, t, c, h, w = x.shape  # B, 5, 64, 64, 64
        H, D = self.heads, self.patch_size * self.patch_size * self.num_feat
        d = D // H
        # q
        h_q = ((h - self.patch_size)//self.stride_q) + 1
        w_q = ((w - self.patch_size)//self.stride_q) + 1
        n_q = h_q * w_q
        # kv
        h_kv = ((h - self.patch_size) // self.stride_kv) + 1
        w_kv = ((w - self.patch_size) // self.stride_kv) + 1
        n_kv = h_kv * w_kv

        q = self.to_q(x.contiguous().view(-1, c, h, w))  # [B*5, 64, 64, 64]
        k = self.to_k(x.contiguous().view(-1, c, h, w))  # [B*5, 64, 64, 64]
        v = self.to_v(x.contiguous().view(-1, c, h, w))  # [B*5, 64, 64, 64]

        unfold_q = F.unfold(q, kernel_size=self.patch_size, padding=0, stride=self.stride_q)   # [B*5, 8*8*64, 8*8]
        unfold_k = F.unfold(k, kernel_size=self.patch_size, padding=0, stride=self.stride_kv)  # [B*5, 8*8*64, 15*15]
        unfold_v = F.unfold(v, kernel_size=self.patch_size, padding=0, stride=self.stride_kv)  # [B*5, 8*8*64, 15*15]

        unfold_q = unfold_q.view(b, t, H, d, n_q)   # [B, 5, H, 8*8*64/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n_kv)  # [B, 5, H, 8*8*64/H, 15*15]
        unfold_v = unfold_v.view(b, t, H, d, n_kv)  # [B, 5, H, 8*8*64/H, 15*15]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 15*15]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 15*15]

        unfold_q = unfold_q.view(b, H, d, t * n_q)   # [B, H, 8*8*64/H, 5*8*8]
        unfold_k = unfold_k.view(b, H, d, t * n_kv)  # [B, H, 8*8*64/H, 5*15*15]
        unfold_v = unfold_v.view(b, H, d, t * n_kv)  # [B, H, 8*8*64/H, 5*15*15]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, 5*8*8, 5*15*15]
        attn = attn * (d ** (-0.5))                              # [B, H, 5*8*8, 5*15*15]
        attn = F.softmax(attn, dim=-1)                           # [B, H, 5*8*8, 5*15*15]

        # print('111111111111', attn.shape)
        # # print("post:", attn)
        # print(torch.max(attn), torch.min(attn))
        # img = attn[0, 0]
        # # img = attn.squeeze(1).transpose(0, 2)
        # img = img / torch.max(img) * 255
        # img = img.detach().cpu().numpy()
        # cv2.imwrite('att_max.png', img)

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))    # [B, H, 5*8*8, 8*8*64/H]
        attn_x = attn_x.view(b, H, t, n_q, d)                    # [B, H, 5, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()      # [B, 5, H, 8*8*64/H, 8*8]
        attn_x = attn_x.view(b * t, D, n_q)                      # [B*5, 8*8*64, 8*8]

        feat = F.fold(attn_x, (h, w), self.patch_size, padding=0, stride=self.stride_q)  # [B*5, 64, 64, 64]

        # inp = torch.ones_like(feat)
        # inp_unfold = F.unfold(inp, self.patch_size, padding=0, stride=self.stride_q)
        # out_mask = F.fold(inp_unfold, (h, w), self.patch_size, padding=0, stride=self.stride_q)
        # feat = feat / out_mask

        out = self.conv(feat).view(x.shape)  # [B, 5, 64, 64, 64]
        out += x  # [B, 5, 64, 64, 64]

        return out

# device = torch.device('cuda:0')
# attn = globalAttention(num_feat=64).to(device)
# x = torch.randn(4, 10, 64, 128, 128).to(device)
# start = datetime.datetime.now()
# for i in range(0, 100):
#     y = attn(x)
#     print(i, y.shape)
# end = datetime.datetime.now()
# print(end - start)

# device = torch.device('cuda:0')
# attn = globalAttention(num_feat=64, patch_size=8, stride_q=8, stride_kv=4, heads=1).to(device)
# x = torch.randn(4, 10, 64, 64, 64).to(device)
# y = attn(x)
# print(y.shape)