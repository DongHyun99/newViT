import torch.nn as nn
import torch

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import math
import matplotlib.pyplot as plt

def learnable_PE(num_patches, dim):
    return nn.Parameter(torch.randn(1, num_patches + 1, dim))

def sinusoidal_PE(num_patches, dim):
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(num_patches+1, dim)
    position = torch.arange(0, num_patches+1).unsqueeze(1)
    denominators = torch.pow(10000.0, 2*torch.arange(0, dim, 2)/dim)
    pe[:, 0::2] = torch.sin(position.float() / denominators)
    pe[:, 1::2] = torch.cos(position.float() / denominators)
    pe = pe.unsqueeze(0)

    return pe

def sinusoidal_PE_2d(num_patches_2d, dim):
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
        
    pe = torch.zeros(num_patches_2d.size()[0], dim)
    denominators = torch.pow(10000.0, 2*torch.arange(0, dim, 2)/dim)
    pe[:, 0::2] = torch.sin(num_patches_2d[:,0].unsqueeze(1).float() / denominators)
    pe[:, 1::2] = torch.cos(num_patches_2d[:,1].unsqueeze(1).float() / denominators)
    pe = pe.unsqueeze(0)

    return pe

def sinusoidal_PE_2d_diagonal(num_patches_2d, diagonal_patches_2d, dim):
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
        
    pe = torch.zeros(num_patches_2d.size()[0], dim)
    denominators = torch.pow(10000.0, 2*torch.arange(0, dim, 2)/dim)
    pe[:, 0::2] = torch.sin((num_patches_2d[:,0].unsqueeze(1).float().float()) / denominators)
    pe[:, 1::2] = torch.cos((num_patches_2d[:,1].unsqueeze(1).float().float()) / denominators) 
    #pe[:, 0::2] = torch.sin(diagonal_patches_2d[:,1].unsqueeze(1).float() / denominators)
    #pe[:, 1::2] = torch.cos(diagonal_patches_2d[:,0].unsqueeze(1).float() / denominators)
    pe = pe.unsqueeze(0)
    
    return pe

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

image_size = 512
patch_size = 16
channels = 3
dim = 512

image_height, image_width = pair(image_size)
patch_height, patch_width = pair(patch_size)

num_patches = (image_height // patch_height) * (image_width // patch_width)
patch_dim = channels * patch_height * patch_width

x_patches = torch.arange(1, image_width//patch_width+1)
y_patches = torch.arange(1, image_height//patch_height+1)
num_patches_2d = torch.cartesian_prod(x_patches, y_patches)
num_patches_2d = torch.concat([torch.tensor([[0,0]]), num_patches_2d], dim=0) # cls token

'''
diagonal_patches_2d = torch.tensor([[0,0]])
y_diagonal_patches = torch.arange(1, image_height//patch_height+1)

for y in y_diagonal_patches:
    x_local_patches = torch.arange(y, y-(image_width//patch_width), -1).unsqueeze(1)
    y_local_patches = torch.arange(y, y+(image_width//patch_width), 1).unsqueeze(1)
    local_patches = torch.concat([x_local_patches, y_local_patches], dim=1)
    diagonal_patches_2d = torch.concat([diagonal_patches_2d, local_patches], dim=0)
    
#diagonal_patches_2d = rearrange(diagonal_patches_2d[1:], '(h w) c -> h w c', w = image_width // patch_width)
#diagonal_patches_2d = torch.transpose(diagonal_patches_2d, 1, 0)
#diagonal_patches_2d = torch.concat([torch.tensor([[0,0]]), rearrange(diagonal_patches_2d, 'h w c -> (h w) c')], dim=0)
'''
#print(f'num_patches: {num_patches}')
#print(f'patch_dim: {patch_dim}')
#pos_embedding = learnable_PE(num_patches, dim)

#############################################################

print('sinusoidal_PE')

pos_embedding = sinusoidal_PE(num_patches, dim)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(pos_embedding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 256))
plt.ylabel('Position')
plt.colorbar()
plt.savefig('PE.png')



pos_encoding = pos_embedding[0,1:,:]
cosSim = nn.CosineSimilarity(dim=0, eps=1e-6)
matrix = torch.zeros((num_patches, num_patches))

for idx1, p1 in enumerate(pos_encoding):
    for idx2, p2 in enumerate(pos_encoding):
        matrix[idx1, idx2] = cosSim(p1, p2)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_similarity.png')

w_patch = image_width // patch_width
h_patch = image_height // patch_height

matrix2 = rearrange(matrix[747], '(w h) -> w h', w=w_patch, h=h_patch)
plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix2.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_similarity_748.png')

#############################################################

print('sinusoidal_PE_2d')

pos_embedding = sinusoidal_PE_2d(num_patches_2d, dim)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(pos_embedding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 256))
plt.ylabel('Position')
plt.colorbar()
plt.savefig('PE_2d.png')

pos_encoding = pos_embedding[0,1:,:]
cosSim = nn.CosineSimilarity(dim=0, eps=1e-6)
matrix = torch.zeros((num_patches, num_patches))

for idx1, p1 in enumerate(pos_encoding):
    for idx2, p2 in enumerate(pos_encoding):
        matrix[idx1, idx2] = cosSim(p1, p2)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_2d_similarity.png')

w_patch = image_width // patch_width
h_patch = image_height // patch_height

matrix2 = rearrange(matrix[748], '(w h) -> w h', w=w_patch, h=h_patch)
plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix2.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_2d_similarity_748.png')

#############################################################
'''
print('sinusoidal_PE_2d_diagonal')

pos_embedding = sinusoidal_PE_2d_diagonal(num_patches_2d, diagonal_patches_2d, dim)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(pos_embedding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 256))
plt.ylabel('Position')
plt.colorbar()
plt.savefig('PE_2d_diagonal.png')

pos_encoding = pos_embedding[0,1:,:]
cosSim = nn.CosineSimilarity(dim=0, eps=1e-6)
matrix = torch.zeros((num_patches, num_patches))

for idx1, p1 in enumerate(pos_encoding):
    for idx2, p2 in enumerate(pos_encoding):
        matrix[idx1, idx2] = cosSim(p1, p2)

plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_2d_diagonal_similarity.png')

w_patch = image_width // patch_width
h_patch = image_height // patch_height

matrix2 = rearrange(matrix[748], '(w h) -> w h', w=w_patch, h=h_patch)
plt.clf()
plt.gca().invert_yaxis()
plt.pcolormesh(matrix2.numpy(), cmap='RdBu')
plt.colorbar()
plt.savefig('PE_2d_diagonal_similarity_748.png')
# '''
#############################################################