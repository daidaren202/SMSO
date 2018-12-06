#coding:utf-8
import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module
import mySMSO_util as utils

x = torch.rand(2,3,4)
print(x[1])
y = utils.calculate_covariance(x)
print(y[1])