#coding:utf-8
import warnings
warnings.filterwarnings('ignore')

class DefaultConfig(object):
	env = 'default'
	batch_size = 32
	print_freq = 1

	test_batch_freq = 1
	test_freq_epoch = 15
	use_gpu = True
	gpu_device = '1'
	num_workers = 8
	
	#csv_path = 'E:/901/code/data/DTD/data.csv'
	#csv_path = '/home/tensorflow/dist4T/jindou/DTD/data.csv'
	csv_path = '/home/tensorflow/jindou/mySMSO/data/DTD/data.csv'
	para_path = './tmp/vgg16based/'
	load_model_path = '/home/tensorflow/jindou/mySMSO/tmp/vgg16based/SMSO_VGG16_1205_02_44_00_68_46.pth'
	#load_model_path = None
	#first = True

	max_epoch = 500
	lr = 1e-4
	lr_decay = 0.95
	weight_decay = 1e-6
	p = 64
	c = 256
	class_num = 47

opt = DefaultConfig()