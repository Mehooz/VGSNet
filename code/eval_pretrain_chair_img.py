"""
    This is the main tester script for point cloud generation evaluation.
    Use scripts/eval_gen_pc_vae_chair.sh to run.
"""
import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from data import PartNetDataset, Tree
from model_part_pc_img import  PartImgDecoder
from torchvision.models import vgg16
import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(1)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.feature_len = 1024
        self.mlp = nn.Linear(512*12*12,1024)


    def forward(self, net):
        net = self.vgg16.features(net)
        net = net.view(net.size(0),-1)
        net = self.mlp(net)

        return net

if __name__ == '__main__':
    device = 'cuda:6'
    decoder = PartImgDecoder(100,3000).to(device)
    decoder.load_state_dict(torch.load('/repository/zhangxc/zhangminghao_tmp_model/part_pc_ae_chair_3000_img/4_net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    net = torch.randn(1, 100).to(device)
    imgs = np.load('/repository/zhangxc/chair_img_3000/172.npz')['image']
    pred = decoder(net,torch.tensor(imgs,dtype=torch.float32).permute(0,3,1,2).to(device))
    print(pred.shape)
    np.savez('../result.npz', parts=pred.cpu().detach().numpy())




