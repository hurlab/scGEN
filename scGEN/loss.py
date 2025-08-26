import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        result = t1 + t2

        result = torch.mean(result)
        return result


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, target, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, tailor_rate=0.9, loss_rate=2, segment_type='tail'):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        mean = mean * 1
        x = target.clone()
        cols = x.shape[1]
        
        # 根据segment_type选择不同的基因段
        segment_size = int(cols * (1 - tailor_rate))  # 10%的基因数量
        
        if segment_type == 'head':
            # 头部10%
            start = 0
            end = segment_size
        elif segment_type == 'middle':
            # 中间10%
            start = (cols - segment_size) // 2
            end = start + segment_size
        elif segment_type == 'random':
            # 随机10%
            start = random.randint(0, cols - segment_size)
            end = start + segment_size
        else:  # 'tail' (默认，原来的方式)
            # 尾部10%
            start = int(cols * tailor_rate)
            end = cols

        slice = x[:, start:end]
        slice = slice * loss_rate
        x[:, start:end] = slice

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)