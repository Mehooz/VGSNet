import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance
from collections import namedtuple
from torchvision.models import vgg16,resnet18,resnet50
import math

class