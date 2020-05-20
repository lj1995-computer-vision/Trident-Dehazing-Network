import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class FFT(torch.nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).cuda(img1.device)
        return nn.L1Loss(size_average=True)(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))#.cuda()