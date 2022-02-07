from basicsr.archs.deblur_arch import Deblur
from basicsr.archs.edvr_arch import EDVR
from scripts.metrics.calculate_flops import get_model_flops
from thop import profile
from ptflops import get_model_complexity_info
import torch

net = Deblur(num_feat=64, num_block=3, spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
# net = EDVR().cuda()
# input_dim = (10, 3, 256, 256)
# flops = get_model_flops(net, input_dim, False)
# print(flops/10**9)

input_dim = torch.randn(1, 10, 3, 256, 256).cuda()
flops, params = profile(model=net, inputs=(input_dim, ))
print(flops/10**9, params/10**6)

# net = Deblur(num_feat=64, num_block=20, spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
# macs, params = get_model_complexity_info(net, (10, 3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
# print('{:<30} {:<8}'.format('Computational complexity:', macs))
# print('{:<30} {:<8}'.format('Number of patameters:', params))