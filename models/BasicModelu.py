#coding:utf-8
import torch
import time
import os

class BasicModule(torch.nn.Module):
	'''
	package the torch.nn.Module and provide the functions 'save' and 'load'
	'''

	def __init__(self):
		super(BasicModule, self).__init__()
		self.model_name = str(type(self)).strip('<>').split('.')[-1][:-1]
		#print('model_name:', self.model_name)

	def load(self, path):
		'''
		load the model according to the path
		'''
		self.load_state_dict(torch.load(path))

	def save(self, acc=None):
		'''
		save the model named name, default setting:model_name+time
		'''
		#if name is None:
		prefix = '/tmp/vgg16based/' + self.model_name + '_'
		#print(os.getcwd())
		name = time.strftime(os.getcwd().replace('\\', '/') + prefix + '%m%d_%H_%M_%S') + '_2048_' + str(acc).replace('.','_') + '.pth'
		torch.save(self.state_dict(), name)
		#print('load name:',name)
		return name