import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision.models import vgg16

from torchvision.models import resnet18


class ImageEncoder(nn.Module):

    def __init__(self,feat_len):
        super(ImageEncoder, self).__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc=nn.Linear(512,512)
        self.mlp1=nn.Linear(512,512)
        self.mlp2=nn.Linear(512,512)
        self.mlp3=nn.Linear(512,feat_len)

    def forward(self, img):
        img_fea = self.resnet(img)

        img_fea=self.mlp1(img_fea)
        img_fea=self.mlp2(img_fea)
        img_fea=self.mlp3(img_fea)
        return img_fea

if __name__=='__main__':
    img=torch.randn(4,3,224,224).to('cuda:3')
    net=ImageEncoder(256).to('cuda:3')
    result=net(g)
