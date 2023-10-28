import torch

from torch import nn

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


def sec2Point(point, elem):
    x, y = point
    return {1:[x-elem, y-elem], 2:[x+elem, y-elem], 3:[x-elem, y + elem], 4:[x+elem, y + elem]}

def divideSection(secions, elem):
    if type(secions[1]) is list:
        return {i:sec2Point(secions[i], elem) for i in range(1,5)}
    return {
        i:divideSection(secions[i], elem) for i in range(1,5)
    }

class getPositionalcoordinate(nn.Module):
    
    def getCoordinates(self, stage_num, initPoint):
        dics, elem = {}, initPoint[0]
        for stage in range(stage_num):
            if stage == 0:
                elem /= 2
                dics = {'stage0' : sec2Point(initPoint, elem)}
                continue    
            stageVal = dics['stage' + str(stage-1)]
            elem /= 2
            dics['stage' + str(stage)] = divideSection(stageVal, elem)
        return dics
    
    def __init__(self, stage_num: int, initPoiont = [1,1]):
        """_summary_

        Args:
            stage_num (int): Stage Number.
            initPoiont (list, optional): _description_. Defaults to [1,1].
        """
        self.stage_num = stage_num
        self.Coordinates = self.getCoordinates(stage_num, initPoiont)

pos = getPositionalcoordinate(stage_num=4)
print(pos.Coordinates)