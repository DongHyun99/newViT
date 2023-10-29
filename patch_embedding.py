import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import torchsummary
import math

from torch import Tensor
from torchvision.transforms import Compose, ToTensor
from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from PIL import Image

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 256, stage: int = 1):
        repeat_num = int(math.log2(img_size//patch_size))
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b (c e) h w -> b c e h w', c=1),
            *[nn.Sequential(
                Rearrange(f'b c e (h1 h2) w -> b c e h2 (h1 w)', h1=2),
                Rearrange(f'b c e h (w1 w2) -> b (c w1) e h w2 ', w1=4))
              for _ in range(0, repeat_num)],
            Rearrange('b c e h w -> b (c h w) e'))
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
            
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        return x

img_size = 512
torchsummary.summary(PatchEmbedding(img_size=512, patch_size=128).cuda(), (3, img_size, img_size), batch_size=4)

'''
patch_size = 1
emb_size = 3
img_size = 512

img = Image.open('data/cat.png').convert('RGB')
transform = Compose([ToTensor()])

img = transform(img)
img = img.unsqueeze(0)

filter = torch.ones((patch_size, patch_size))
filter = filter.unsqueeze(0).unsqueeze(0)
filter = repeat(filter, 'o i w h ->(dim1 o) i w h', dim1 = emb_size)

x = F.conv2d(img, filter, stride = (patch_size, patch_size), groups=emb_size)
x = rearrange(x, 'b (c e) h w -> b c e h w', c=1)

value = 2

for idx in range(1,value+1):
    x = rearrange(x, f'b c e (h1 h2) w -> b c e h2 (h1 w)', h1=2) # 1 cycle
    x = rearrange(x, f'b c e h (w1 w2) -> b (c w1) e h w2 ', w1=4) # new channel

x = rearrange(x, 'b c e h w -> h (c w) (e b)') * 255.0 / patch_size**2
print(x.shape)
x = np.array(x.detach(), dtype=np.uint8)
x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
cv2.imwrite('32.png', x)
# '''