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
from model_part_vst_fea import  PartEncoder,PartImgDecoder
from torchvision.models import vgg16
from image_encoder import ImageEncoder
import torch.nn as nn
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(1)

def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc

class PartNetGeoDataset(torch.utils.data.Dataset):
    def __init__(self, root, object_list, use_local_frame):
        self.root = root
        self.use_local_frame = use_local_frame

        if isinstance(object_list, str):
            with open(os.path.join(root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]

        else:
            self.object_names = object_list

    def __getitem__(self, index):
        theta_x = -np.pi / 6
        theta_y = np.pi / 4
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        views = np.random.randint(1, 25)
        fn = os.path.join(self.root, self.object_names[index]+f'_{views}.npz')
        data = np.load(fn)['parts']

        fn_img = os.path.join(self.root.replace('_geo','_img'), self.object_names[index] + f'_{views}.npz')
        img_set = np.load(fn_img)['image']
        # idx = np.random.randint(min(data.shape[0],img_set.shape[0]))
        idx = np.random.randint(data.shape[0])
        # idx = 0
        pts = data[idx, :, :]
        pts = normal_pc(pts)
        # pts = np.dot(pts, Ry)
        # pts = np.dot(pts, Rx)
        pts = torch.tensor(pts, dtype=torch.float32)
        img = img_set[idx,:,:,:]
        # img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img,dtype = torch.float32)
        if self.use_local_frame:
            pts = pts - pts.mean(dim=0)
            pts = pts / pts.pow(2).sum(dim=1).max().sqrt()
        return (pts, img ,self.object_names[index], idx)

    def __len__(self):
        return len(self.object_names)





if __name__ == '__main__':
    device = 'cuda:0'
    theta_x = -np.pi / 6
    theta_y = np.pi / 4
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])

    #img_encoder=ImageEncoder(512).to(device)
    encoder = PartEncoder(256,False).to(device)
    decoder = PartImgDecoder(512,3000).to(device)
    encoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet-master/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_multi_2_parts_train/6_net_part_pc_encoder.pth',map_location={'cuda:0':device}))
    decoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet-master/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_multi_2_parts_train/6_net_part_pc_decoder.pth',map_location={'cuda:0':device}))

    id = '2420_1'
    # img = np.load(f'/home/zhangxc/tmp/structurenet/data/partnetdata/chair_img_3000/{id}.npz')['image'][0:1]
    # pcs = np.load(f'/home/zhangxc/tmp/structurenet/data/partnetdata/chair_geo_3000/{id}.npz')['parts'][0:1]
    img = np.load(f'/repository/zhangxc1/chair_img_new/{id}.npz')['image'][1:2]
    pcs = np.load(f'/repository/zhangxc1/chair_geo_new/{id}.npz')['parts'][1:2]
    # img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).to(device).permute(0,3,1,2)

    # pcs = np.dot(pcs, Ry)
    # pcs = np.dot(pcs, Rx)
    pcs = torch.tensor(pcs, dtype=torch.float32).to(device)

    # pcs = torch.randn(1, 3000,3).cuda()
    # img = torch.randn(1,3,224,224).cuda()

    net = encoder(pcs,img)
    p,i = decoder(net)

    print(p.shape)
    print(i.shape)
    cv2.imwrite(f'../data/results/{id}.jpg',i.permute(0,2,3,1).squeeze(0).cpu().detach().numpy()*255)
    np.savez(f'../data/results/{id}.npz', parts=p.cpu().detach().numpy())




