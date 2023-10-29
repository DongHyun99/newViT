import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn as nn

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
    """
        Inputs:
            stage_num - Number of Transformer block
        """
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

    def __init__(self, stage_num, initPoiont = [1,1]):
        self.stage_num = stage_num
        self.Coordinates = self.getCoordinates(stage_num, initPoiont)
    
    #def sinusoidal_PE()

pos = getPositionalcoordinate(stage_num=4)
print(pos.Coordinates)