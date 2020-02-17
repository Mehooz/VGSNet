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
from torchvision.models import vgg16
import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(1)


if __name__ == '__main__':
    device = 'cuda:3'
    models = utils.get_model_module('model_part_pc')
    encoder = models.PartEncoder(100).to(device)
    decoder = models.PartDecoder(100,1000).to(device)
    decoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair/194_net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    encoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair/194_net_part_pc_encoder.pth',map_location={'cuda:0':device}))

    # net = torch.randn(1, 100).to(device)
    with torch.set_grad_enabled(False):
        pts = np.load('/home/zhangxc/tmp/structurenet/data/partnetdata/chair_geo_3000/172.npz')['parts'][0:2]
        # pts = pts[np.newaxis, :]
        pts = torch.tensor(pts, dtype=torch.float32).to(device)
        net = encoder(pts)
        pred = decoder(net)
        print(pred.shape)
        np.savez('../result_all.npz', parts=pred.cpu().detach().numpy())




