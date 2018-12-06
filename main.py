#coding:utf-8
from config import opt
import numpy as np
import os
import torch
#import models
from data.dataset import DTD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.SMSO_VGG16 import SMSO_VGG16
import torchvision.models as models
import scipy.io as sio
import time
# from torchnet import meter
# from utils import Visualizer
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

def train():
	#step1: configure model
	model = SMSO_VGG16()


	#print(model)
	if opt.load_model_path: model.load(opt.load_model_path)
	else:
    		vgg16 = models.vgg16(pretrained=True)
    		pre_dict = vgg16.state_dict()
    		#print(type(pre_dict))
    		model_dict = model.state_dict()
    		pre_dict = {k:v for k, v in pre_dict.items() if k in model_dict}
    		model_dict.update(pre_dict)
    		model.load_state_dict(model_dict)

	if opt.use_gpu: model.cuda()
	#for i in range(20):
	#	test(model)
	#return model

	#step2: data
	train_dataset = DTD(opt.csv_path, train=True)
	print('train_dataset_len: ',len(train_dataset.imgs))
	# for i in train_data.imgs:
	# 	print(i)
	train_dataloader = DataLoader(train_dataset, opt.batch_size, 
		shuffle=True, num_workers=opt.num_workers)

	#step3: criterion and optimizer
	criterion = torch.nn.CrossEntropyLoss()
	lr = opt.lr
	weight_decay = opt.weight_decay
	#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

	start = time.time()
	#step5: train
	for epoch in (range(opt.max_epoch)):
	#for epoch in range(1):
		running_loss = torch.tensor(0.0).cuda()
		n = 0
		for i, (data, label) in (enumerate(train_dataloader)):
			n += 1
			break
			input = Variable(data)
			#print(input.shape)
			target = Variable(label)
			optimizer.zero_grad()
			if opt.use_gpu:
				input = input.cuda()
				target = target.cuda()
			output = model(input)
			mid = torch.softmax(output,1)
			#print('mid\n',mid[0])
			_, pred = torch.max(mid.data, 1)
			#print(pred, target)


			#if n>5: break
			loss = criterion(output, target)
			#print('output',output)
			#print('target',target)
			loss.backward()
			optimizer.step()
			#print('loss, loss.data[0]:', loss,loss.data[0])
			running_loss += loss.data[0]
			if (i+1)%opt.print_freq == 0 :
				#print('runningloss, print_freq:',running_loss, opt.print_freq)
				print('epoch:%d %5d loss:%.3f' % (epoch+1, (i+1)*opt.batch_size, running_loss/opt.print_freq))
				running_loss = 0.0
			
			#break
		if (epoch+1)%opt.test_freq_epoch == 0:
			#model.save()
			end = time.time()
			interval = end-start
			print('interval:%.2fm per %d epoches' % (interval/60, opt.test_freq_epoch))
			start = time.time()
			test()

	return model

def val(model, dataloader):
	val_data = DTD(opt.csv_path, val=True)
	# print(len(train_data))
	pass

def test():

	#step2: data
	model = SMSO_VGG16()
	model.load(opt.load_model_path)
	if opt.use_gpu: model.cuda()

	test_dataset = DTD(opt.csv_path, test=True)
	print('test_dataset_len: ',len(test_dataset.imgs))
	test_dataloader = DataLoader(test_dataset, opt.batch_size, 
		shuffle=False, num_workers=opt.num_workers)


	#step5: test
	corrects = 0.0
	cor_sum = 0.0
	total = 0.0
	acc = 0.0
	#acc_l = []
	for i, (data, label) in (enumerate(test_dataloader)):
		#print(label)
		#continue
		input = Variable(data)
		#print(input.shape)
		target = Variable(label)
		if opt.use_gpu:
			input = input.cuda()
			target = target.cuda()
		output = model(input)
		_, pred = torch.max(output.data, 1)
		print('pred   :',pred.data)
		print('target :',target.data)
		corrects += torch.eq(pred, target).sum()
		
		
		if (i+1)%opt.test_batch_freq == 0 :
			#print('dataindex:%5d cor:%d total: %d' % ((i+1)*opt.batch_size, corrects,(opt.test_batch_freq * opt.batch_size)))
			#print('dataindex:%5d acc:%.2f%%' % ((i+1)*opt.batch_size, (corrects*100.0)/(1.0*opt.test_batch_freq * opt.batch_size * 1.0)))
			cor_sum += corrects
			total += (opt.batch_size*opt.test_batch_freq)
			corrects = 0.0
	acc = 1.0*cor_sum.cpu().numpy()/total
	acc = round(100*acc,2)
	print('cor_sum:%d total:%d acc:%.2f' % (cor_sum, total, acc))
	#model.save(acc)


def help():
	pass

def init_W_beta_gama():
	W_path = opt.para_path + 'W.mat'
	if not os.path.exists(W_path):
		W = np.random.rand(opt.c,opt.p) * np.sqrt(1/opt.p)
		W = W.astype(np.float32)
		sio.savemat(W_path,{'W':W})

	beta_path = opt.para_path + 'beta.mat'
	if not os.path.exists(W_path):
		beta = np.zeros((opt.p, 1))
		beta = beta.astype(np.float32)
		sio.savemat(beta_path, {'beta' : beta})

	gama_path = opt.para_path + 'gama.mat'
	if not os.path.exists(gama_path):
		gama = 0.01 * np.random.rand(opt.p, 1)
		gama = gama.astype(np.float32)
		sio.savemat(gama_path, {'gama' : gama})

def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device
	#torch.cuda.set_device(opt.gpu_device)
	init_W_beta_gama()
	#model = train()
	#model = SMSO_VGG16()
	#if opt.use_gpu: model.cuda()
	#model.load(opt.load_model_path)
	test()

if __name__ == '__main__':
	# dirs = (dir(opt))
	# for i in dirs:
	# 	if hasattr(opt, i):
	# 		print(i,getattr(opt,i))
	main()



		