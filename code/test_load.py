import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, num_point*3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

    def forward(self, net):
        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net
if __name__ == '__main__':
    model = PartDecoder(100,100)
    # model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)
    m = torch.load('../data/models/part_pc_ae_chair/194_net_part_pc_decoder.pth')
    model.load_state_dict(m,strict=True)