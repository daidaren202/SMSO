#coding:utf-8
import torch
import torchvision.models as models
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

a = 67.5

s = str(a).replace('.', '_')
print(s)