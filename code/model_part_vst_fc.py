"""
    This file defines part point cloud VAE/AE model.
"""

import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance
from collections import namedtuple
from torchvision.models import vgg16,resnet18,resnet50
from loss import *
from projection import *
import math

class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu

class PartImgSampler(nn.Module):

    def __init__(self, feat_len):
        super(PartImgSampler, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.mlp = nn.Linear(1000, feat_len)

    def forward(self, img):
        net = self.resnet(img)
        net = self.mlp(net)
        return net

class PartEncoder(nn.Module):

    def __init__(self, feat_len, probabilistic=False):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)
        self.mlp = nn.Linear(512,512)

        self.pc_sampler = PartFeatSampler(feature_size=feat_len, probabilistic=probabilistic)
        self.img_sampler = PartImgSampler(feat_len=feat_len)

    def forward(self, pc, img):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net_pc = self.pc_sampler(net)
        net_img = self.img_sampler(img)
        out = torch.cat((net_img,net_pc[:,0:256]),1)
        out = self.mlp(out)

        return out,net_pc[:,256:512]


class PartImgDecoder(nn.Module):
    def __init__(self, feat_len, num_point):
        super(PartImgDecoder,self).__init__()
        self.num_point = num_point

        self.pc_deconv1 = nn.ConvTranspose2d(4, 8, (3, 3))
        self.pc_deconv2 = nn.ConvTranspose2d(8, 16, (3, 3))
        self.pc_deconv3 = nn.ConvTranspose2d(16, 32, (5, 5))
        self.pc_mlp  =nn.Linear(32*16*16,9000)

        self.img_deconv1 = nn.ConvTranspose2d(8,16,(5,5))
        self.img_deconv2 = nn.ConvTranspose2d(16,32, (5,5))
        self.img_deconv3 = nn.ConvTranspose2d(32,48, (5,5))
        self.img_deconv4 = nn.ConvTranspose2d(48, 48, (5, 5))
        self.img_deconv5 = nn.ConvTranspose2d(48, 48, (7, 7))
        self.img_deconv6 = nn.ConvTranspose2d(48, 48, (7, 7))
        self.img_deconv7 = nn.ConvTranspose2d(48, 48, (7, 7))
        self.img_deconv8 = nn.ConvTranspose2d(48, 48, (7, 7))
        self.mlp = nn.Linear(512,512)
        self.img_mlp1 = nn.Linear(256,1024)
        self.img_mlp2 = nn.Linear(1024,2048)

        self.chamferLoss = ChamferDistance()
        self.mse = nn.MSELoss()
        self.gdt = grid_dist(64, 64)

    def forward(self, net):

        net = self.mlp(net)
        img = net[:,0:256]
        pc = net[:,256:512]
        pc = pc.view(net.shape[0], -1, 8, 8)

        pc = self.pc_deconv1(pc)
        pc = self.pc_deconv2(pc)
        pc = self.pc_deconv3(pc)
        pc = pc.view(-1,32*16*16)
        pc = self.pc_mlp(pc)
        pc = pc.view(net.shape[0],3000,3)


        img = self.img_mlp1(img)
        img = self.img_mlp2(img)
        img = img.view(net.shape[0],-1,16,16)
        img = self.img_deconv1(img)
        img = self.img_deconv2(img)
        img = self.img_deconv3(img)
        img = self.img_deconv4(img)
        img = self.img_deconv5(img)
        img = self.img_deconv6(img)
        img = self.img_deconv7(img)
        img = self.img_deconv8(img)
        img = img.view(net.shape[0],3,224,224)

        return pc, img

    def loss(self, pred_pc, gt_pc, pred_img, gt_img,n_views,batch_size,device,n_points):
        dist1, dist2 = self.chamferLoss(pred_pc, gt_pc)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        avg_loss = loss.mean() * 3000
        mse_loss =self.mse(pred_img,gt_img)

        views_x = torch.rand(batch_size, 4) * 2 * math.pi
        views_y = torch.rand(batch_size, 4) * 2 * math.pi

        return avg_loss, mse_loss

        # proj_loss= torch.tensor(0.0).to(device)
        #
        # for i in range(n_views):
        #     pred_rot=world2cam(pred_pc,views_x[:,i],views_y[:,i],batch_size,device,n_points)
        #     pred_persp=perspective_transform(pred_rot,device,batch_size)
        #     pred_proj=cont_proj(pred_persp,64,64,device)
        #
        #     gt_rot = world2cam(gt_pc, views_x[:,i], views_y[:,i], batch_size,device, n_points)
        #     gt_persp = perspective_transform(gt_rot,device, batch_size)
        #     gt_proj = cont_proj(gt_persp, 64, 64,device)
        #
        #     #bceloss, min_dist, min_dist_inv = get_loss_proj(pred_proj,gt_proj,device,'bce_prob',1.0,None,self.gdt.to(device))
        #     bceloss = get_loss_proj(pred_proj, gt_proj, device, 'bce_prob', 1.0, None,
        #                                                     self.gdt.to(device))
        #     proj_loss+=torch.mean(bceloss)
        #
        #     #proj_loss+=1e-4*torch.mean(min_dist).item()
        #     #proj_loss+=1e-4*torch.mean(min_dist_inv).item()
        #
        # proj_loss=proj_loss/n_views * n_points
        #
        #
        # return avg_loss, mse_loss ,proj_loss

if __name__ == '__main__':
    # pe = PartEncoder(256).to('cuda')
    # net = pe(torch.randn(1,3000,3).to('cuda'),torch.randn(1,3,224,224).to('cuda'))
    pid = PartImgDecoder(512,3000).to('cuda')
    # pid (net)
    loss = pid.loss(torch.randn(1,3000,3).to('cuda'),torch.randn(1,3000,3).to('cuda'),torch.randn(1,3,224,224).to('cuda'),torch.randn(1,3,224,224).to('cuda'),4,1,'cuda',3000)
    print(loss)
