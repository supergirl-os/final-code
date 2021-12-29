#实现感知器准则、最小平方误差准则和Fisher准则，并针对数字5和8的识别进行评估实验；
#2021.4.8

import numpy as np
import random
import time
import tensorflow as tf
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
def get_data():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# print(x_train[1])
	x_Train = x_train.reshape(60000, 784).astype('float32')
	x_Test = x_test.reshape(10000, 784).astype('float32')

	x_train, x_test = x_Train / 255.0, x_Test / 255.0
	x_train_part = np.concatenate((x_train[y_train == 5], x_train[y_train == 8]), axis=0)
	y_train_part = np.concatenate((y_train[y_train == 5], y_train[y_train == 8]), axis=0)
	x_test_part = np.concatenate((x_test[y_test == 5], x_test[y_test == 8]), axis=0)
	y_test_part = np.concatenate((y_test[y_test == 5], y_test[y_test == 8]), axis=0)
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_labels[i]-5==test_predict[i] or test_labels[i]-7==test_predict[i]):
			correct +=1
	return correct
def f1_score_help(test_labels,test_predict):
	l = len(test_labels)
	for i in range(l):
		if(test_labels[i]==8):
			test_labels[i]=1
		else:
			test_labels[i]=0
	return f1_score(test_labels, test_predict, average='weighted')

class Perceptron(object):

	def __init__(self):
		self.learning_step = 0.00001  #迁移率
		self.max_iteration = 10000    #最大迭代次数

	def predict_(self,x):
		wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
		return int(wx>0)

	def train(self,features,labels):
		self.w = [0.0]*(len(features[0])+1)  #加1是要考虑常系数b的存在
		correct_count = 0
		time = 0  #这里的time是设置次数

		while time<self.max_iteration:
			index = random.randint(0,len(labels) - 1)
			x = list(features[index])#某一个样本向量转成列表
			x.append(1.0)   #加1.0  同理也是要考虑常系数b的存在
			#print(x)
			#y = 0.67*labels[index] - 4.33
			y=0.7*labels[index]-4.5
			wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
			#print(wx)
			#print(labels[index])
			#print(y)
			if wx*y>0:
				correct_count +=1
				if correct_count>self.max_iteration:
					#print("已经大于了")
					break
				continue

			for i in range(len(self.w)):
				self.w[i] += self.learning_step*(y*x[i])    #更新

	def predict(self,features):
		labels = []
		for feature in features:
			x = list(feature)
			x.append(1)
			labels.append(self.predict_(x))
		return labels
if __name__ == '__main__':
	print("Start read data")
	time_1 = time.time()

	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_Train = x_train.reshape(60000, 784).astype('float32')
	x_Test = x_test.reshape(10000, 784).astype('float32')

	x_train, x_test = x_Train / 255.0, x_Test / 255.0
	x_train_part = np.concatenate((x_train[y_train == 5], x_train[y_train == 8]), axis=0)
	y_train_part = np.concatenate((y_train[y_train == 5], y_train[y_train == 8]), axis=0)
	x_test_part = np.concatenate((x_test[y_test == 5], x_test[y_test == 8]), axis=0)
	y_test_part = np.concatenate((y_test[y_test == 5], y_test[y_test == 8]), axis=0)

	train_features=x_train_part
	test_features=x_test_part
	train_labels=y_train_part
	test_labels=y_test_part

	time_2 = time.time()
	print("read data cost ",time_2-time_1,' second','\n')

	print('Start training')
	p = Perceptron()
	p.train(train_features,train_labels)

	time_3 = time.time()
	print("training cost ",time_3-time_2,' second','\n')

	print('Start predicting')
	test_predict = p.predict(test_features)
	time_4 = time.time()
	print('predicting cost',time_4-time_3,' second','\n')
	# print(test_labels)
	# print(test_predict)
	correct=accuracy(test_labels,test_predict)
	score=correct/len(test_labels)
	print(test_labels.shape)
	print("The accuracy score is ",score)
	print("The f1_score is",f1_score_help(test_labels, test_predict))
	print("混淆矩阵为：",confusion_matrix(test_labels, test_predict))

