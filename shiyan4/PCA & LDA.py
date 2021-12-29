#使用的分类器是knn分类器
#

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier


def show_mnist():
    mnist = datasets.load_digits()
    X = mnist.data
    y = mnist.target

    pca = decomposition.PCA(n_components=3)
    new_X = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
    plt.show()


def PCA_test(K,x_train,y_train,x_test,y_test):
    if K==1:
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(x_train, y_train)
        accr = knn_clf.score(x_test, y_test)
        #print("PCA的预测准确度为：", accr)
        return x_train.shape[1], accr
    else:
        pca = PCA(K)
        pca.fit(x_train)
        X_train_reduction = pca.transform(x_train)
        print(X_train_reduction.shape[1])
        X_test_reduction = pca.transform(x_test)
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train_reduction, y_train)
        accr = knn_clf.score(X_test_reduction, y_test)
        print("PCA的预测准确度为：", accr)
        return X_train_reduction.shape[1], accr
def LDA_test(x_train,y_train,x_test,y_test):
    n_class=len((np.unique(y_train)))
    print(n_class)
    X1 = np.zeros(n_class)  # 测试维数
    Y1 = np.zeros(n_class)

    i=0
    for n in range(1,n_class):
        print("n = ", n)
        lda = LDA(n_components=n)
        lda.fit(x_train, y_train)
        X_train_reduction = lda.transform(x_train)
        #print(X_train_reduction)
        X_test_reduction = lda.transform(x_test)
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train_reduction, y_train)
        accr = knn_clf.score(X_test_reduction, y_test)
        print("LDA的预测准确度为：", accr)
        X1[i]=X_train_reduction.shape[1]
        Y1[i]=accr
        i+=1

    knn_clf_t = KNeighborsClassifier()
    knn_clf_t.fit(x_train, y_train)
    accr = knn_clf_t.score(x_test, y_test)
    print("LDA的预测准确度为：", accr)
    X1[i]=10
    Y1[i]=accr
    return X1,Y1





if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_Train = x_train.reshape(60000, 784).astype('float32')
    x_Train1=x_Train[:20000]
    y_Train1=y_train[:20000]
    x_Train2=x_Train[:40000]
    y_Train2 =y_train[:40000]
    print(x_Train1.shape)
    x_Test = x_test.reshape(10000, 784).astype('float32')

    rates=[1,0.9,0.8,0.7,0.6]
    x1=np.zeros(5)
    y1=np.zeros(5)
    x2 = np.zeros(5)
    y2 = np.zeros(5)
    x3 = np.zeros(5)
    y3 = np.zeros(5)
    i=0
    for rate in rates:
        x1[i],y1[i]=PCA_test(rate,x_Train,y_train,x_Test,y_test)#60000
        x2[i], y2[i] = PCA_test(rate, x_Train2, y_Train2, x_Test, y_test)#40000
        x3[i], y3[i] = PCA_test(rate, x_Train1, y_Train1, x_Test, y_test)#20000
        i+=1
        # 绘图评估PCA准确率
    plt.title('PCA-accuracy with different scale')
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.plot(x1, y1, marker='o', color='green', label='PCA_ACCR_60000')
    plt.plot(x2, y2, marker='+', color='blue', label='PCA_ACCR_40000')
    plt.plot(x3, y3, marker='*', color='red', label='PCA_ACCR_20000')
    plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
    plt.grid()
    plt.show()

    # X1 = np.zeros(10)#测试维数
    # Y1 = np.zeros(10)
    #
    # j=0
    # X1, Y1 = LDA_test(x_Train, y_train, x_Test, y_test)
    # X2,Y2=LDA_test(x_Train2, y_Train2, x_Test, y_test)
    # X3,Y3 = LDA_test(x_Train1, y_Train1, x_Test, y_test)
    # print(X1)
    # print(Y1)
    #
    # plt.title('LDA-accuracy with different scale')
    # plt.plot(X1, Y1, marker='*', color='red', label='LDA_ACCR_60000')
    # plt.plot(X2, Y2, marker='o', color='green', label='PCA_ACCR_40000')
    # plt.plot(X3, Y3, marker='+', color='blue', label='PCA_ACCR_20000')
    # plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
    # plt.grid()
    # plt.show()