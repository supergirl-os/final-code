import numpy as np
import time
import tensorflow as tf
from sklearn.cluster import KMeans
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_labels[i]-4==test_predict[i] or test_labels[i]-8==test_predict[i]):
			correct +=1
	return correct
if __name__ == '__main__':
    print("Start read data")
    time_1 = time.time()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_Train = x_train.reshape(60000, 784).astype('float32')
    x_Test = x_test.reshape(10000, 784).astype('float32')
    x_train_part = np.concatenate((x_Train[y_train == 5], x_Train[y_train == 8]), axis=0)
    y_train_part = np.concatenate((y_train[y_train == 5], y_train[y_train == 8]), axis=0)
    x_test_part = np.concatenate((x_Test[y_test == 5], x_Test[y_test == 8]), axis=0)
    y_test_part = np.concatenate((y_test[y_test == 5], y_test[y_test == 8]), axis=0)
    x_Train = x_train_part
    y_train = y_train_part
    x_Test = x_test_part
    y_test = y_test_part


    print(x_Train.shape)
    time_2 = time.time()

    print("read data cost ",time_2-time_1,' second','\n')

    print('Start training')
    k = 2
    clf = KMeans(n_clusters=2)
    clf.fit(x_Train)
    #无监督学习
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    print(centers)
    print(labels)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    label=clf.predict(x_Test)
    print(label)
    print(y_test)
    correct = accuracy(y_test, label)
    score = correct / len(label)
    print("The accuracy score is ", score)