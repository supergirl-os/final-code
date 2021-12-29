#实现感知器准则、最小平方误差准则和Fisher准则，并针对数字5和8的识别进行评估实验；
#2021.4.9
#准确率只有60%
import numpy as np
import time
import math
import tensorflow as tf
from sklearn.metrics import f1_score
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_labels[i]==test_predict[i]):
			correct +=1
	return correct
class my_Fisher:
    def __init__(self):
        # 无用数据
        self.learning_step = 0.00001  # 迁移率
        self.max_iteration = 50000  # 最大迭代次

    def cal_matrix(self,x,m):
        sw = np.zeros((x.shape[1], x.shape[1]))
        for index in x:
            t = index - m
            sw += t * t.transpose()
        return sw
    def train(self,x_train,y_train):
        x1=x_train[y_train==5]
        x2=x_train[y_train==8]
        m1=np.mean(x1,axis=0)
        m2=np.mean(x2,axis=0)
        sw1=self.cal_matrix(x1,m1)
        sw2=self.cal_matrix(x2,m2)
        s_w=sw1+sw2
        print(s_w)
        s_w_inv=np.linalg.pinv(s_w)
        # u, s, v = np.linalg.svd(s_w)  # 奇异值分解
        # s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
        self.w= np.dot(s_w_inv, m1 - m2)

        self.y0=(self.w.T.dot(m1.reshape(-1,1))+self.w.T.dot(m2.reshape(-1,1)))/2
        self.center1 = sum([self.w[j] * m1[j] for j in range(len(self.w))])
        self.center2 = sum([self.w[j] * m2[j] for j in range(len(self.w))])

    def predict_(self,x_test):


        wx = sum([self.w[j] * x_test[j] for j in range(len(self.w))])
        l_1=0.45
        l_2=0.05
        pw1=0.35
        pw2=0.05
        center=l_1*self.center1+l_2*self.center2-math.log(pw1/pw2)/40

        if(wx<-7e-5):
            return 5
        else:
            return 8
    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            #x.append(1)
            labels.append(self.predict_(x))
        return labels




if __name__ == '__main__':
    print("Start read data")
    time_1 = time.time()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_Train = x_train.reshape(60000, 784).astype('float32')
    x_Test = x_test.reshape(10000, 784).astype('float32')
    #x_train, x_test = x_Train / 255.0, x_Test / 255.0
    x_train_part = np.concatenate((x_Train[y_train == 5], x_Train[y_train == 8]), axis=0)
    y_train_part = np.concatenate((y_train[y_train == 5], y_train[y_train == 8]), axis=0)
    x_test_part = np.concatenate((x_Test[y_test == 5], x_Test[y_test == 8]), axis=0)
    y_test_part = np.concatenate((y_test[y_test == 5], y_test[y_test == 8]), axis=0)
    print(x_train_part.shape)
    x_train=x_train_part
   # x_train = np.c_[x_train_part, np.ones((len(y_train_part), 1))]
    y_train = y_train_part

    time_2 = time.time()
    print("read data cost ",time_2-time_1,' second','\n')

    print('Start training')
    fisher=my_Fisher()
    fisher.train(x_train,y_train)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict=fisher.predict(x_test_part)
    print(test_predict)
    print(y_test_part)
    time_4 = time.time()
    print('predicting cost', time_4 - time_3, ' second', '\n')
    correct = accuracy(y_test_part, test_predict)
    score = correct / len(y_test_part)
    print("The accuracy score is ", score)
    print("The f1_score is", f1_score(y_test_part, test_predict, average='weighted'))


