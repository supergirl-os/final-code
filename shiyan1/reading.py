import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import _pickle as pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import csv

from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
plt.show()












#下载mnist.npz并存放于用户/.keras/datasets/下
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# #print(x_train[1])
# x_Train = x_train.reshape(60000, 784).astype('float32')
# x_Test = x_test.reshape(10000, 784).astype('float32')
#
# x_train, x_test = x_Train / 255.0, x_Test / 255.0
# print(x_train[0])
# print(x_train.shape)
# print(len(y_train))
# print(type(x_train))
# x_train_part = np.concatenate((x_train[y_train == 5],x_train[y_train == 8]), axis=0)
# y_train_part = np.concatenate((y_train[y_train == 5],y_train[y_train == 8]), axis=0)
# x_test_part = np.concatenate((x_test[y_test == 5],x_test[y_test == 8]), axis=0)
# y_test_part = np.concatenate((y_test[y_test == 5],y_test[y_test == 8]), axis=0)


# with open('D:/本科课内学习资料集锦/大二下学习资料包/模式识别/MNIST/MNIST/mnist_train.csv','r') as f:
#     reader = csv.reader(f)
#     result = list(reader)
#     print(result[1])



# train_images=np.zeros((3000,784))
# train_labels=np.array(3000)
# test_images=np.zeros((3000,784))
# test_labels=np.array(3000)
# for i in range(len(y_train)):
#     if(y_train[i]==5 or y_train[i]==8):
#         train_images=x_train[i]
#         train_labels=y_train[i]
# for j in range(len(y_test)):
#     if(y_test[j]==5 or y_test[j]==8):
#         test_images=x_test[j]
#         test_labels=y_test[j]

#
# plt.figure(figsize=(5, 5))
# class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])  # 坐标轴
#     plt.yticks([])
#     plt.grid(True, axis='x')  # 网格
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_name[y_train_part[i]])  # 去对应的标签
# plt.show()  # 显示图片
#

# model = DecisionTreeClassifier(criterion="entropy")
# model.fit(x_train,y_train)
# z=model.predict(x_test)
# print('准确率：',np.sum(z==y_test)/z.size)
# #学习后识别99到105六张图片并给出预测
# model.predict(mnist.data[99:105])
# #实际的99到105代表的数
# mnist.target[99:105]
# #显示99到105数字图片
# plt.subplot(321)
# plt.imshow(mnist.images[99],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.subplot(322)
# plt.imshow(mnist.images[100],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.subplot(323)
# plt.imshow(mnist.images[101],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.subplot(324)
# plt.imshow(mnist.images[102],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.subplot(325)
# plt.imshow(mnist.images[103],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.subplot(326)
# plt.imshow(mnist.images[104],cmap=plt.cm.gray_r,interpolation='nearest')
# from six import StringIO
# import pandas as pd
# x = pd.DataFrame(x_train)
# with open("tree.dot", 'w') as f:
#      f = export_graphviz(model, feature_names = x.columns, out_file = f)
