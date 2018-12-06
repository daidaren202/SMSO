#coding:utf-8
import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module

def sub_cov(x):
	'''
	x:n*c
	y:c*c
	'''
	Y = torch.rand((x.size(1), x.size(1)),dtype=torch.float32)
	#print('x shape and x****************\n',x.shape,x[0])
	mean = x.sum(0)/x.size(0)
	xx = x-mean
	xxt = torch.transpose(xx,1,0)
	#print('xx',xx)
	#print('xxt',xxt)
	y = (1.0/(x.size(0)-1))*torch.matmul(xxt,xx)
	#y = torch.matmul(xxt,xx)
	#print(y)
	#print('sub Y:*******************\n',y[0])
	#print('caonima*******lalala****\n',y.shape)
	return y


def calculate_covariance(x):
	'''
	x:batch_size*n*c 196 256
	y = cov(x)	batch_size*c*c 256 256
	'''
	#print(' x shape lalala:',x.shape)
	y = torch.rand((x.size(0),x.size(2),x.size(2)),dtype=torch.float32).cuda()
	for i in range(x.size(0)):
		y[i] = sub_cov(x[i])
	#print('cal y shape',y.shape)
	#print('cov Y**************************************\n',y)
	return y


def sub_pv(X, W):
	#print('sub_cov X shape/*/*/*/*/*/*/*\n',X.shape)
	c = W.size(0)
	p = W.size(1)
	#print('cp',c,p)
	Y = torch.zeros((p,1),dtype=torch.float32)
	for j in range(p):
		#print('pv')
		wjt = W[:,j].view(1,c)
		
		#print('wjt shape:', wjt.shape)
		wj = W[:,j].view(c,1)
		#print('wt shape:', wj.shape)
		#print('here++++++++++++++++',wjt.shape,X.shape)
		tmp = torch.mm(wjt,X)
		
		#print('tmp shape:',tmp.shape)
		Y[j] = torch.mm(tmp, wj)
		#print('Y[i] shape:', Y[j].shape)
		#print(Z[i].shape)
	#print('Y shape:',Y.shape)
	#print('Y:', Y)
	return Y
	pass

def pv(Y, W):
	'''
	pv operation
	zi = wiTYwi
	'''
	#print('Y,W:\n',Y,W)
	#print('pv:',Y.device,W.device)
	batch_size = Y.size(0)
	c = W.size(0)
	p = W.size(1)
	#print('W shape',W.shape)
	Z = torch.zeros((batch_size,p,1),dtype=torch.float32).cuda()
	for i in range(batch_size):
		
		Z[i] = sub_pv(Y[i], W)
		#print('tmp:', tmp)
		Z[i] = torch.sqrt(Z[i])
		#print('Z[i]:', Z[i])
		#break

	Z = Z.view(batch_size,-1)
	#print('pv Z shape:', Z.shape)
	#print('Z:', Z)
	return Z
