#实现感知器准则、最小平方误差准则和Fisher准则，并针对数字5和8的识别进行评估实验；
#2021.4.9
import numpy as np
import time
import tensorflow as tf

from sklearn.metrics import f1_score
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_labels[i]==test_predict[i]):
			correct +=1
	return correct


class my_LMSE:
    def __init__(self):
        #无用数据
        self.learning_step = 0.00001  # 迁移率
        self.iteration = 250  # 最大迭代次
    def train(self,x_train,y_train):
        #计算伪逆
        tempo=np.matmul(x_train.transpose(), x_train)
        XX = np.matmul(np.linalg.pinv(tempo), x_train.transpose())#计算伪逆
        #初始化：b(1) > 0, k = 1, 0，c≤1
        b=np.ones(len(y_train))
        k=1
        c=1
        iteration = 20
        t=0#迭代次数
        # 计算权重w(k) = x  # b(k)，及误差e(k)=xw(k)-b(k)
        self.w = np.matmul(XX, b)
        e = np.matmul(x_train, self.w) - b
        #如果e(k)的所有分量都为负数，则算法结束，无解；
        #如果e(k)=0，则算法结束，输出权重w(k)；
        #否则，算法继续

        while 1:
            temp = min(e)
            temp1 = max(e)
            if 0 > temp > -1e-4:
                temp = 0
            if temp > 1e-3:
                deltab = e + abs(e)
                b = b + c * deltab
                self.w = self.w + c * np.matmul(XX, deltab)
                e = np.matmul(x_train, self.w) - b
            else:
                if 1e-4 > temp >= 0:
                    break
                else:
                    # 线性不可分
                    if temp1 < 0:
                        flag = 1
                        break
                    else:
                        # 趋近时迭代
                        deltab = e + abs(e)
                        b = b + c * deltab
                        self.w = self.w + c * np.matmul(XX, deltab)
                        e = np.matmul(x_train, self.w) - b
                        t = t + 1
                        if t >= self.iteration:
                            break
        #print(self.w, '\n', e)
    def predict_(self,x_test):
        wx = sum([self.w[j] * x_test[j] for j in range(len(self.w))])
        if(wx>0):
            return 5
        else:
            return 8

    def predict(self, features):
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
    #print(x_train_part.shape)
    # 每个向量加一 ,将训练样本符号规范化，得x
    x_train = np.c_[x_train_part,np.ones((len(y_train_part),1))]
    x_temp=x_train[y_train_part==8]
    x_temp=-x_temp
    x_train=np.concatenate((x_train[y_train_part==5],x_temp),axis=0)
    y_train=y_train_part

    time_2 = time.time()
    print("read data cost ",time_2-time_1,' second','\n')

    print('Start training')
    lmse=my_LMSE()
    lmse.train(x_train,y_train)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict=lmse.predict(x_test_part)

    time_4 = time.time()
    print('predicting cost', time_4 - time_3, ' second', '\n')
    correct = accuracy(y_test_part, test_predict)
    score = correct / len(y_test_part)
    print("The accuracy score is ", score)
    print("The f1_score is", f1_score(y_test_part, test_predict,average='weighted'))
#错误：没有明确5，8对应哪个域；求广义逆矩阵