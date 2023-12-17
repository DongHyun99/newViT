import torch
import torch.nn as nn
from einops import rearrange
import math


class PatchDivideLayer(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.divideLayer = nn.ConvTranspose2d(
            in_channels = embed_dim,
            out_channels = out_dim,
            kernel_size=2, 
            stride=2
            )
        self.divideLayer = nn.ConvTranspose1d(
            in_channels = embed_dim,
            out_channels = out_dim,
            kernel_size=4, 
            stride=4,
            #padding=3
            )
    
    def forward(self, x, selcted_infos):
        """_summary_
         Args:
            x (_type_): feature map [b, patch_num, d]
            selcted_info (_type_): contains dividing sections information [b, patch_num]
             1 : 쪼갤 patch, 0 : 안쪼갤 patch
         Output : [b, (patch_num-2) + 2*4, d]
        """
        # 1. patch 모두 쪼개기
        print('origin : ', x.shape)
        B = x.shape[0]
        x_divide = rearrange(x, 'b n c -> b c n')
        x_divide = self.divideLayer(x_divide)
        x_divide = rearrange(x_divide, 'b c n -> b n c')

        print('1. 패치 쪼개기 : ', x_divide.shape)
        out = []
        for b in range(B):  # batch 사이즈 마다 수행
            batch_x = x[b]      # x*4
            tmp = []
            for secIdx in range(len(batch_x)):
                if selcted_infos[b, secIdx] == 1:    # 해당 section을 쪼갤시
                    tmp.append(x_divide[b, 4*secIdx: 4*secIdx + 4])
                else:                              
                    tmp.append(x[b, secIdx].unsqueeze(dim=0))
                    
                print('[', b,'batch] ', secIdx, ' : ', selcted_infos[b,secIdx], tmp[-1].shape)
            out.append(torch.concat(tmp, dim=0).unsqueeze(dim=0))
            
        return torch.concat(out, dim = 0)
        
layer = PatchDivideLayer(512, 512)
x = torch.rand(3,4, 512)
sections = torch.Tensor([
    [True, True, False, False],
    [True, False, False, True],
    [False, True, True, False]
])
'''x = torch.rand(3,10,512)
sections = torch.Tensor([
    [True, True, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, True, True, False],
    [False, True, True, False, False, False, False, False, False, False]
])'''
layer(x, sections)