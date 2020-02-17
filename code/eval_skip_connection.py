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
# from model_part_vst_fea import  PartEncoder,PartImgDecoder
import model_part_vst_fea
import model_part_vst
from torchvision.models import vgg16
import torch.nn as nn
from emd_loss import EMD
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
torch.set_num_threads(1)


class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P

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
        fn = os.path.join(self.root, self.object_names[index]+'.npz')
        data = np.load(fn)['parts']
        fn_img = os.path.join(self.root.replace('_geo','_img'), self.object_names[index] + '.npz')
        fn_gt = os.path.join(self.root.replace('_geo_3000','_geo'), self.object_names[index] + '.npz')
        gt = np.load(fn_gt)['parts']
        img_set = np.load(fn_img)['image']
        # idx = np.random.randint(min(data.shape[0],img_set.shape[0]))
        idx = np.random.randint(data.shape[0])

        pts = data[0:1, :, :]
        gt_pts = gt[0:1, : ,:]
        # pts = normal_pc(pts)
        pts = np.dot(pts, Ry)
        pts = np.dot(pts, Rx)
        pts = torch.tensor(pts, dtype=torch.float32)

        gt_pts = np.dot(gt_pts, Ry)
        gt_pts = np.dot(gt_pts, Rx)
        gt_pts = torch.tensor(gt_pts, dtype=torch.float32)

        img = img_set[0:1,:,:,:]
        # img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img,dtype = torch.float32)
        if self.use_local_frame:
            pts = pts - pts.mean(dim=0)
            pts = pts / pts.pow(2).sum(dim=1).max().sqrt()
        return (pts, img ,gt_pts ,self.object_names[index])

    def __len__(self):
        return len(self.object_names)

if __name__ == '__main__':
    device = 'cuda:6'
    theta_x = -np.pi / 6
    theta_y = np.pi / 4
    # Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    # Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])

    data_path = '/repository/zhangxc1/chair_geo_new'
    test_dataset = 'test_no_other_less_than_10_parts.txt'

    encoder_fea = model_part_vst_fea.PartEncoder(256,False).to(device)
    decoder_fea =model_part_vst_fea. PartImgDecoder(512,3000).to(device)
    #encoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_parts/579_net_part_pc_encoder.pth',map_location={'cuda:0':device}))
    #decoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_parts/579_net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    encoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_parts/485_net_part_pc_encoder.pth',map_location={'cuda:0':device}))
    decoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_fea_kl0.01_parts/485_net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    #encoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_table_3000_vst_fea_kl0.01_parts/906_net_part_pc_encoder.pth',map_location={'cuda:0':device}))
    #decoder_fea.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_table_3000_vst_fea_kl0.01_parts/906_net_part_pc_decoder.pth',map_location={'cuda:0':device}))
    # encoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_vase_3000_vst_fea_kl0.01_parts/594_net_part_pc_encoder.pth',map_location={'cuda:0':device}))
    # decoder.load_state_dict(torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_vase_3000_vst_fea_kl0.01_parts/594_net_part_pc_decoder.pth',map_location={'cuda:0':device}))

    encoder_no = model_part_vst.PartEncoder(256, False).to(device)
    decoder_no = model_part_vst.PartImgDecoder(512, 3000).to(device)
    encoder_no.load_state_dict(
        torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_4/508_net_part_pc_encoder.pth',
                   map_location={'cuda:0': device}))
    decoder_no.load_state_dict(
        torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_chair_3000_vst_4/508_net_part_pc_decoder.pth',
                   map_location={'cuda:0': device}))


    test_dataset = PartNetGeoDataset(data_path, test_dataset, use_local_frame=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, \
                                                   shuffle=True, collate_fn=utils.collate_feats, num_workers=4)

    train_batches = enumerate(test_dataloader, 0)
    cd = EMD()
    losses_fea = []
    losses_no = []
    for train_batch_ind, batch in train_batches:
        pts_s =  batch[0]
        imgs_s = batch[1]
        gt_s = batch[2]
        for i in range(0,len(pts_s)):
            with torch.no_grad():
                imgs = imgs_s[i].permute(0, 3, 1, 2).to(device)
                net_no = encoder_no(pts_s[i].to(device), imgs)
                pred_pc_no, pred_img_no = decoder_no(net_no)

                net_fea = encoder_fea(pts_s[i].to(device), imgs)
                pred_pc_fea, pred_img_fea = decoder_fea(net_fea)

                #np.savetxt('')

                cd_loss_fea = cd(pred_pc_fea,gt_s[i].to(device))
                cd_loss_no = cd(pred_pc_no,gt_s[i].to(device))

                total_loss_fea = cd_loss_fea.mean()/3000
                losses_fea.append(total_loss_fea.item())

                total_loss_no = cd_loss_no.mean() / 3000
                losses_no.append(total_loss_no.item())

                print('fea:', np.array(losses_fea).mean())
                print('no:', np.array(losses_no).mean())
    print('final:',np.array(losses_fea).mean())







