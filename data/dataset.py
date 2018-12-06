#coding:utf-8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt 
import warnings
warnings.filterwarnings('ignore')

class DTD(data.Dataset):

	def __init__(self, root, transforms=None, train=False, val=False, test=False):
		'''
		get the data and split them into train, val and test subset;
		'''
		imgs = []
		#stage = '1' if train else '2' if val else '3' if test else '4'
		#val -> train ,discard the val
		stage = '1' if train else '1' if val else '3' if test else '4'
		print('stage: ',('train' if stage=='1' else 'val' 
			if stage=='2' else 'test' if stage=='3' else 'UNKNOWNSTAGE'))
		#print('csvroot:',root)
		f = open(root, 'r')

		f.readline()
		lines = f.readlines()#
		for line in lines:
			#print('i am in')
			#print(line)
			contents = line.strip('\n\r').split(',')
			# if contents[-1] == stage:
			# 	item = {'path':contents[0], 'label':int(contents[-2])-1}
			# 	imgs.append(item)
			if stage == '1':
				#print('i am stage1')
				#print(contents[-1],type(contents[-1]))
				if contents[-1] == '1' or contents[-1] == '2':
					#print('i am stage1')
					item = {'path':contents[0], 'label':int(contents[-2])-1}
					imgs.append(item)	
			elif stage == '3':
				#print('i am stage3')
				if contents[-1] == '3' :
					item = {'path':contents[0], 'label':int(contents[-2])-1}
					imgs.append(item)							
		
		#print('imgs len:',len(imgs))
		#shuffle imgs
		# np.random.permutation(100)
		# imgs = np.random.permutation(imgs)	
		self.imgs = imgs
		print(len(imgs))
		if transforms is None:
			normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
									std = [0.229,0.224,0.225])
			if test :
				self.transforms = T.Compose([
					T.Scale(256),
					T.CenterCrop(224),
					T.ToTensor(),
					normalize
					])
			else:
				self.transforms = T.Compose([
					T.Scale(256),
					T.RandomSizedCrop(224),
					T.RandomHorizontalFlip(),
					T.ToTensor(),
					normalize
					])
		

	def __getitem__(self, index):
		'''
		return a picture according to the given index once time;
		'''
		#img_path = os.path.join('DTD',self.imgs[index]['path'])
		img_path = opt.csv_path[:-8]+self.imgs[index]['path']
		#print(img_path)
		data = Image.open(img_path)
		data = self.transforms(data)
		label = self.imgs[index]['label']
		#print('getitem data--------------------------------------------\n',data)
		return data, label

	def __len__(self):
		return len(self.imgs)

if __name__ == '__main__':
	root = './DTD/data.csv'
	dtd = DTD(root,train=True)
	print(dtd.imgs)