import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class BRELU(torch.nn.Module):
    def __init__(self):
        super(BRELU, self).__init__()

    def forward(self, img1, img2):
        img_brelu=img1.clamp(0,1)
        bs,c,a,b=img1.shape
        return torch.log((img1 - img_brelu).abs() + 1).sum()/bs/c/a/b