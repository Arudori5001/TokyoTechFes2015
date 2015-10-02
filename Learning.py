#coding: utf-8

import chainer
from chainer import FunctionSet, optimizers
import chainer.functions as F
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, relpath
import glob
import pickle
from PIL import Image
import sys
import time
import random


from ChainerModel import ChainerModel
from Record import Record


def forward(model, x_data, ratio=0.1,train=True):
	'''
	@summary: フィードフォワードを行う
	'''

	x = chainer.Variable(x_data, volatile=False)
	h = F.max_pooling_2d(F.relu(model.conv1(x)), 3, stride=2)
	h = F.average_pooling_2d(F.relu(model.conv2(h)), 3, stride=2)
	h = F.average_pooling_2d(F.relu(model.conv3(h)), 3, stride=2)
	h = F.dropout(F.relu(model.fl5(h)),train=train,ratio=ratio)
	y = model.fl6(h)

	return y


def setup_model(gpu_id, n_channel, n_output):
	model = FunctionSet(conv1=F.Convolution2D(n_channel,32,5,pad=2),
		conv2=F.Convolution2D(32,32,5,pad=2),
		conv3=F.Convolution2D(32,64,5,pad=2),
		fl5=F.Linear(960, 64),
		fl6=F.Linear(64, n_output))
	#optimizer = optimizers.MomentumSGD(lr=1e-03)
	optimizer = optimizers.AdaGrad()
	optimizer.setup(model.collect_parameters())

	mlp = ChainerModel(model,optimizer,forward_function=forward)
	return mlp


def img_to_record(img, label):
	shape = (3, img.size[0], img.size[1])
	
	pix = img.load()
	inp = np.zeros(shape, dtype=np.float32)
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			inp[:,i,j] = pix[i,j]		
	
	inp = np.array(inp, dtype=np.float32)
	return Record(inp, label)
	

def get_records(dirs):
	records = []
	for i,dr in enumerate(dirs):
		for fl in glob.glob(join(dr, "*")):
			img = Image.open(fl)
			img = img.resize((36,46))
			record = img_to_record(img, i)
			records.append(record)
			
	return records


def show_figure(train_accs,test_accs):
	'''
	@summary: 学習データとテストデータの誤差の推移をグラフ描画する
	'''
	
	plt.figure(figsize=(8,6))
	plt.plot(range(len(train_accs)), train_accs)
	plt.plot(range(len(test_accs)), test_accs)
	plt.legend(["train_accs","test_accs"],loc=4)
	plt.title("Accuracy of baseball players recognition.")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.ylim([0.,1.1])
	plt.plot()
	plt.show()


def main():
	dirs = glob.glob("dataset/*")
	
	n_channel = 3
	n_output = len(dirs)
	n_train = sys.argv[1]
	n_test = sys.argv[2]
	max_epoch = 100
	look_back = 10
	prop_increase = 0.01
	batchsize = 10
	gpu_id = -1
	allow_dropout = False
	dropout_ratio = 0.0


	records = get_records(dirs)
	np.random.shuffle(records)
	inputs = [r.input for r in records]
	labels = [r.label for r in records]
	inputs = np.array(inputs, dtype=np.float32)
	labels = np.array(labels, dtype=np.int32)
	train_inputs, test_inputs, _ = np.split(inputs, [n_train, n_train+n_test])
	train_labels, test_labels, _ = np.split(labels, [n_train, n_train+n_test])
	
	mlp = setup_model(gpu_id, n_channel, n_output)

	train_accs,test_accs = mlp.learn(train_inputs, train_labels, test_inputs, test_labels,
		max_epoch, look_back, prop_increase, batchsize, gpu_id,
		allow_dropout=allow_dropout,dropout_ratio=dropout_ratio)
	
	with open("model.pkl","wb") as f:
		pickle.dump(mlp.model, f)
		
	show_figure(train_accs,test_accs)
	
	print("")
	print("----------")
	print("finished")



if __name__ == "__main__":
	main()