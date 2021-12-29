import numpy as np
from numpy import *
import operator
import time
import tensorflow as tf
from sklearn.model_selection import cross_val_score


array([0.9502, 0.96565, 0.96495])
def KNN(test_data, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#dataSet.shape[0]表示的是读取矩阵第一维度的长度，代表行数
    # distance1 = tile(test_data, (dataSetSize,1)) - dataSet#欧氏距离计算开始
    # print("dataSetSize:")
    # print(dataSetSize)
    distance1 = np.tile(test_data, (dataSetSize,1))-dataSet#tile函数在行上重复dataSetSizec次，在列上重复1次
    # print("distance1.shape")
    # print(distance1.shape)
    distance2 = distance1**2 #每个元素平方
    distance3 = distance2.sum(axis=1)#矩阵每行相加
    distances4 = distance3**0.5#欧氏距离计算结束
    # print(distances4[53843])
    # print(distances4[38620])
    # print(distances4[16186])
    sortedDistIndicies = distances4.argsort() #返回从小到大排序的索引
    classCount=np.zeros((10), np.int32)#10是代表10个类别
    for i in range(k): #统计前k个数据类的数量
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] += 1
    max = 0
    id = 0
    #print(classCount.shape[0])
    # print(classCount.shape[1])
    #print(classCount)
    for i in range(classCount.shape[0]):
        if classCount[i] >= max:
            max = classCount[i]
            id = i
    #print(id)

    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)#从大到小按类别数目排序
    return id


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
    time_2 = time.time()

    print("read data cost ", time_2 - time_1, ' second', '\n')

    print('Start training')
    testRatio = 1  # 取数据集的前0.1为测试数据,这个参数比重可以改变
    train_row = x_train_part.shape[0]  # 数据集的行数，即数据集的总的样本数
    test_row = x_test_part.shape[0]
    testNum = int(test_row * testRatio)
    errorCount = 0  # 判断错误的个数
    for i in range(testNum):
        result = KNN(x_test_part[i],x_train_part, y_train_part, 30)
        # print('返回的结果是: %s, 真实结果是: %s' % (result, train_y[i]))

        #print(result, y_test_part[i])
        if result != y_test_part[i]:
            errorCount += 1.0  # 如果mnist验证集的标签和本身标签不一样，则出错
    error_rate = errorCount / float(testNum)  # 计算出错率
    acc = 1.0 - error_rate

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (error_rate))
    print("\nthe total accuracy rate is: %f" % (acc))