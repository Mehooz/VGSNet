"""
    This file defines part point cloud VAE/AE model.
"""
import math
import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance
from loss import *
from projection import *
from collections import namedtuple
from torchvision.models import vgg16
from torchvision.models import resnet50

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

        self.sampler = PartFeatSampler(feature_size=feat_len, probabilistic=probabilistic)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net


class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, num_point * 3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

        self.chamferLoss = ChamferDistance()

    def forward(self, net):
        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net

    def loss(self, pred, gt):
        dist1, dist2 = self.chamferLoss(pred, gt)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        avg_loss = loss.mean() * 1000

        return avg_loss


class PartImgDecoder(nn.Module):
    def __init__(self, feat_len, num_point):
        super(PartImgDecoder,self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, (5, 5))
        self.deconv2 = nn.ConvTranspose2d(64, 64, (5, 5))
        self.deconv3 = nn.ConvTranspose2d(64, 32, (5, 5))
        self.mlp4 = nn.Linear(32*16*16,num_point*3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

        self.chamferLoss = ChamferDistance()

        self.vgg16 = vgg16(pretrained=True)
        self.mlp = nn.Linear(512 * 12 * 12, 1024)

        self.gdt=grid_dist(64,64)

    def forward(self, net, img):
        img_fea = self.vgg16.features(img)
        img_fea = img_fea.view(img.size(0), -1)
        img_fea = self.mlp(img_fea)

        net = self.mlp1(net)
        net = self.mlp2(net)
        net = self.mlp3(net)
        net = net + img_fea
        net = net.view(net.shape[0],-1,4,4)
        net = self.deconv1(net)
        net = self.deconv2(net)
        net = self.deconv3(net)
        net = net.view(-1,512*16)
        net = self.mlp4(net)
        net = net.view(-1, self.num_point, 3)

        return net

    def loss(self, pred, gt,n_views,batch_size,device,n_points):
        dist1, dist2 = self.chamferLoss(pred, gt)
        views_x = torch.rand(batch_size, 4) * 2 * math.pi
        views_y = torch.rand(batch_size, 4) * 2 * math.pi


        proj_loss=0

        for i in range(n_views):
            pred_rot=world2cam(pred,views_x[:,i],views_y[:,i],batch_size,device,n_points)
            pred_persp=perspective_transform(pred_rot,device,batch_size)
            pred_proj=cont_proj(pred_persp,64,64,device)

            gt_rot = world2cam(gt, views_x[:,i], views_y[:,i], batch_size,device, n_points)
            gt_persp = perspective_transform(gt_rot,device, batch_size)
            gt_proj = cont_proj(gt_persp, 64, 64,device)

            #bceloss, min_dist, min_dist_inv = get_loss_proj(pred_proj,gt_proj,device,'bce_prob',1.0,None,self.gdt.to(device))
            bceloss = get_loss_proj(pred_proj, gt_proj, device, 'bce_prob', 1.0, None,
                                                            self.gdt.to(device))
            proj_loss+=torch.mean(bceloss).item()

            #proj_loss+=1e-4*torch.mean(min_dist).item()
            #proj_loss+=1e-4*torch.mean(min_dist_inv).item()

        proj_loss=proj_loss/n_views



        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        avg_loss = loss.mean()*1000

        return avg_loss+proj_loss*1000
