import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def sinusoidal_PE_2d(num_patches_2d, dim):
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
        
    pe = torch.zeros(num_patches_2d.size()[0], dim, device=0)
    denominators = torch.pow(10000.0, 2*torch.arange(0, dim, 2)/dim)
    pe[:, 0::2] = torch.sin(num_patches_2d[:,0].unsqueeze(1).float() / denominators)
    pe[:, 1::2] = torch.cos(num_patches_2d[:,1].unsqueeze(1).float() / denominators)
    pe = pe.unsqueeze(0)

    return pe

def coordinateList(stage: int, focus: np.ndarray = np.array([1.0, 1.0]), radius: float = 1.0):
    
    if stage == 0:
        return np.expand_dims(focus, 0)
    
    else:
        radius /= 2
        return np.concatenate((coordinateList(stage-1, np.array([focus[0]-radius, focus[1]-radius]), radius),
        coordinateList(stage-1, np.array([focus[0]+radius, focus[1]-radius]), radius),
        coordinateList(stage-1, np.array([focus[0]-radius, focus[1]+radius]), radius),
        coordinateList(stage-1, np.array([focus[0]+radius, focus[1]+radius]), radius)))

def getPositionalCoordinate(stage_num: int, focus: list = [1.0, 1.0]):
    
    coordinate = {}
    
    for stage in range(stage_num+1):
        coordinate[stage] = coordinateList(stage, focus)
            
    return coordinate      

       
def heatmap(hit_position: dict, position_value: dict, stage:int = 1, count: int = 0):
    
    coordinate = np.array([])
    
    for position in range(count*4, count*4+4):
        if position in hit_position[stage]:
            coordinate = np.append(coordinate, position_value[stage][position])
            
        else:
            value = heatmap(hit_position, position_value, stage+1, position)
            coordinate = np.append(coordinate, value)
            
    return coordinate

# test
hit_position = {1:[0,2], 2:[4,7,14,15], 3:[20,21,22,23,24,25,26,27,48,49,50,51,52,53,54,55]}
pos = getPositionalCoordinate(stage_num=3)
hit_position = rearrange(heatmap(hit_position, pos), '(v c) -> v c', c=2)
print(hit_position)

# visualizing
plt.figure(figsize=(10, 10))
plt.xlim(0.0, 2.0)
plt.ylim(0.0, 2.0)
plt.gca().invert_yaxis()
for idx, val in enumerate(hit_position):
    plt.scatter(val[0], val[1], label=idx)
plt.legend()
plt.savefig('coordinate.png')

# positional encoding
hit_position = torch.Tensor(hit_position)
print(hit_position)
print(sinusoidal_PE_2d(hit_position, 512))