import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np

from torch import Tensor
from torchvision.transforms import Compose, ToTensor
from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from PIL import Image

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 3, img_size: int = 512, dimension: bool = False): # 768
        self.patch_size = patch_size
        super().__init__()
        
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),)
            #Rearrange('b e (h) (w) -> b (h w) e'),)
                    
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

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

value = 1

for idx in range(1,value+1):
    x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} (h1 h2) w -> b {" ".join(f"c{c}" for c in range(idx))} h2 (h1 w)', h1=2) # 1 cycle
    if idx != value:
        x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} h (w1 w2) -> b {" ".join(f"c{c}" for c in range(idx))} w1 h w2 ', w1=4) # new channel

for idx in range(value, 1, -1):
    x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} h w -> b {" ".join(f"c{c}" for c in range(idx-1))} h ({f"c{idx-1}"} w)') # channel reduction

print(x.size())

x = rearrange(x, 'b c h w -> h w (c b)') * 255.0 / patch_size**2
x = np.array(x.detach(), dtype=np.uint8)
x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
cv2.imwrite('32.png', x)

'''
x = rearrange(x, 'b c (h1 h2) w -> b c h2 (h1 w)', h2=img_size//patch_size//2) # 1 cycle
x = rearrange(x, 'b c1 h (c2 w2) -> b c1 c2 h w2 ', w2 =img_size//patch_size//2) # new channel w1
x = rearrange(x, 'b c1 c2 (h1 h2) w -> b c1 c2 h2 (h1 w)', h2=img_size//patch_size//2//2) # 2cycle
x = rearrange(x, 'b c1 c2 h w -> b c1 h (c2 w)') # channel reduction
'''

'''
patch_size = 1
emb_size = 712
img_size = 512

img = Image.open('data/cat.png').convert('RGB')
transform = Compose([ToTensor()])

img = transform(img)
img = img.unsqueeze(0)

filter = torch.ones((3, patch_size, patch_size))
filter = filter.unsqueeze(0).unsqueeze(0)
filter = repeat(filter, 'o i w h ->(dim1 o) i w h', dim1 = emb_size)

x = F.conv2d(img, filter, stride = (patch_size, patch_size))

value = 4

for idx in range(1,value+1):
    x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} (h1 h2) w -> b {" ".join(f"c{c}" for c in range(idx))} h2 (h1 w)', h1=2) # 1 cycle
    if idx != value:
        x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} h (w1 w2) -> b {" ".join(f"c{c}" for c in range(idx))} w1 h w2 ', w1=4) # new channel

for idx in range(value, 1, -1):
    x = rearrange(x, f'b {" ".join(f"c{c}" for c in range(idx))} h w -> b {" ".join(f"c{c}" for c in range(idx-1))} h ({f"c{idx-1}"} w)') # channel reduction

print(x.size())

x = rearrange(x, 'b c h w -> h w (c b)') * 255.0 / patch_size**2
x = np.array(x.detach(), dtype=np.uint8)
x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
cv2.imwrite('32.png', x)
'''