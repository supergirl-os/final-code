from sklearn.cluster import KMeans
import numpy as np
import time
import tensorflow as tf
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_predict[i]==test_labels[i]):
			correct +=1
	return correct


if __name__ == '__main__':
    print("Start read data")
    time_1 = time.time()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_Train = x_train.reshape(60000, 784).astype('float32')
    x_Test = x_test.reshape(10000, 784).astype('float32')

    print(x_Train.shape)
    time_2 = time.time()

    print("read data cost ",time_2-time_1,' second','\n')

    print('Start training')
    K=2
    kmeans = KMeans(n_clusters=10, random_state=0).fit(x_Train)
    #print(kmeans.centers_)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    print(y_train)
    print(kmeans.labels_)
    test_predict=kmeans.predict(x_Test)


    print(y_test)
    print(test_predict)

    time_4 = time.time()
    print('predicting cost', time_4 - time_3, ' second', '\n')
    correct = accuracy(y_test, test_predict)
    print(correct)
    score = correct / len(y_test)
    print("The accuracy score is ", score)