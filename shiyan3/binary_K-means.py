import matplotlib.pyplot as plt
#50%
import numpy as np
from scipy.spatial.distance import cdist
import time
import tensorflow as tf
from sklearn.metrics import silhouette_score
def accuracy(test_labels,test_predict):
	l=len(test_labels)
	correct=0
	for i in range(l):
		if(test_labels[i]-5==test_predict[i] or test_labels[i]-7==test_predict[i]):
			correct +=1
	return correct


class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max=100):
        I = np.eye(self.n_clusters)
        centers = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(iter_max):
            prev_centers = np.copy(centers)
            D = cdist(X, centers)
            cluster_index_num = np.argmin(D, axis=1)
            cluster_index = I[cluster_index_num]
            centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis=0) / np.sum(cluster_index, axis=0)[:, None]
            if np.allclose(prev_centers, centers):
                break
        self.centers = centers
        return centers

    def predict(self, X):
        D = cdist(X, self.centers)
        return np.argmin(D, axis=1)

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
    x_Train=x_train_part
    y_train=y_train_part
    x_Test=x_test_part
    y_test=y_test_part
    print(x_Train.shape)
    time_2 = time.time()

    print("read data cost ",time_2-time_1,' second','\n')

    print('Start training')
    K=2
    kmeans=KMeans(K)
    kmeans.fit(x_Train)
    #print(kmeans.centers_)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict=kmeans.predict(x_Test)

    print(test_predict)
    #print(test_predict.shape)
    print(y_test)
    print(y_test.shape)

    time_4 = time.time()
    print('predicting cost', time_4 - time_3, ' second', '\n')
    correct = accuracy(y_test, test_predict)
    print(correct)
    score = correct / len(y_test)
    print("The accuracy score is ", score)
