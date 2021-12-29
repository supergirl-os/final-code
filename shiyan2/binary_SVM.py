#2-1PR
#2021.4.9
#实现支持向量机算法（可使用开源代码），针对MNIST中5和8进行评估实验。
import numpy as np
from sklearn import metrics
from sklearn import svm
import tensorflow as tf
import time
from sklearn.metrics import f1_score
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
    time_2 = time.time()
    print("read data cost ", time_2 - time_1, ' second', '\n')

    print('Start training')
    svm=svm.SVC(gamma='scale')
    svm.fit(x_train_part,y_train_part)
    time_3 = time.time()
    print("training cost ", time_3 - time_2, ' second', '\n')

    print('Start predicting')
    svm.predict(x_test_part)
    y_pred = svm.predict(x_test_part)
    time_4 = time.time()
    print('predicting cost', time_4 - time_3, ' second', '\n')
    #  用测试集计算准确率
    a=metrics.accuracy_score(y_test_part,y_pred)
    print("The accuracy score is ",a)
    print("The f1_score is", f1_score(y_test_part, y_pred, average='weighted'))