"""
    This is the main tester script for point cloud generation evaluation.
    Use scripts/eval_gen_pc_vae_chair.sh to run.
"""
import os
import sys
import json
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from torchvision.models import vgg16
import torch.nn as nn
from config import add_eval_args
from data import PartNetDataset, Tree
from image_encoder import ImageEncoder

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(1)

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--num_gen', type=int, default=20, help='how many shapes to generate?')
eval_conf = parser.parse_args()


if __name__ == '__main__':
    device = 'cuda:3'
    models = utils.get_model_module('model_pc')
    conf = torch.load('/home/zhangxc/tmp/structurenet-master/data/models/pc_ae_chair_3000_test/conf.pth')
    # load object category information
    Tree.load_category_info(conf.category)
    # merge training and evaluation configurations, giving evaluation parameters precendence
    conf.__dict__.update(eval_conf.__dict__)
    conf.num_point = 1000
    conf.exp_name = 'pc_ae_chair_3000_test'
    conf.hidden_size=256
    conf.feature_size=256

    img_encoder=ImageEncoder(256).to(device)
    img_encoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet-master/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_multi_2_parts_img_encoder/net_img_encoder.pth',map_location={'cuda:0':device}))
    decoder = models.RecursiveDecoder(conf).to(device)
    decoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_multi_2/net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    img_encoder.eval()
    decoder.eval()
    # net = torch.randn(1, 100).to(device)
    with torch.set_grad_enabled(False):
        imgs_1 = np.load('/home/zhangxc/tmp/structurenet/data/partnetdata/chair_img_3000/173.npz')['image'][4:8]
        imgs_2 = np.load('/home/zhangxc/tmp/structurenet/data/partnetdata/chair_img_3000/186.npz')['image'][1:2]
        imgs = np.concatenate((imgs_2,imgs_1),axis=0)
        print(imgs.shape)
        imgs = torch.tensor(imgs, dtype=torch.float32).permute(0,3,1,2).to(device)
        root_codes=img_encoder(imgs)
        root_code=root_codes[2]
        pred = decoder.decode_structure(z=root_code.unsqueeze(0), max_depth=conf.max_tree_depth)
        output_filename = os.path.join(conf.result_path,'pc_ae_chair_image_encoder_test', 'object-result.json')
        PartNetDataset.save_object(obj=pred, fn=output_filename)
        with open(output_filename) as f:
            data=json.load(f)
            np.save('result', data['geo'])





