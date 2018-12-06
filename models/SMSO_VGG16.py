#coding:utf-8
from torch import nn
from .BasicModelu import BasicModule
import torchvision.models as models
import utils.mySMSO_util as utils
import torch
import scipy.io as sio
from config import opt
from torch.autograd import Variable

class SMSO_VGG16(BasicModule):
    '''
    model from paper 'Statistically-motivated Second-order Pooling' based on vgg16
    '''

    def __init__(self, num_class=47):
        super(SMSO_VGG16, self).__init__()
        self.model_name = 'SMSO_VGG16'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
         )
        self.halfConv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
     
        self.bn_ = nn.BatchNorm1d(opt.p)
        self.fc_ = nn.Linear(opt.p, opt.class_num)

        tmp = sio.loadmat(opt.para_path + 'W.mat')['W']
        self.W = Variable(torch.from_numpy(tmp), requires_grad=True)

        #tmp = sio.loadmat(opt.para_path + 'beta.mat')['beta']
        #self.beta = Variable(torch.from_numpy(tmp), requires_grad=True)

        #tmp = sio.loadmat(opt.para_path + 'gama.mat')['gama']
        #self.gama = Variable(torch.from_numpy(tmp), requires_grad=True)
        if opt.use_gpu:
            self.W = self.W.cuda()
            #self.beta = self.beta.cuda()
            #self.gama = self.gama.cuda()
 
    def forward(self, x):
        x = self.features(x)
        x = self.halfConv (x)
        #print('forward---------------\n',x[0][0])

        x = x.view(x.size(0),x.size(1),-1)
        x = torch.Tensor.permute(x,0,2,1)
        x = utils.calculate_covariance(x) #batch_size*256*256
        x = utils.pv(x, self.W)

        x = x.view(x.size(0),-1)
        x = self.bn_(x)
        x = self.fc_(x)
        #print('return---------------\n',x)
        return x    


if __name__ == '__main__':
    x = torch.rand(2,3,4)
    print(x[1])
    y = utils.calculate_covariance(x)
    print(y[1])

