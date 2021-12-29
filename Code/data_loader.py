from scipy.io import loadmat
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


# 将标签进行转化，相当于to_categorical，将标签转为向量
def mapping(src):
    dst = np.zeros(10)
    if src == 0 or src ==10:
        dst[0] = 1
    elif src == 1:
        dst[1] = 1
    elif src == 2:
        dst[2] = 1
    elif src == 3:
        dst[3] = 1
    elif src == 4:
        dst[4] = 1
    elif src == 5:
        dst[5] = 1
    elif src == 6:
        dst[6] = 1
    elif src == 7:
        dst[7] = 1
    elif src == 8:
        dst[8] = 1
    else:
        dst[9] = 1
    return dst


def data_loader():
    m_train = loadmat("data/train.mat")
    m_test = loadmat("data/test.mat")
    train_data, train_labels = m_train['X'], m_train['y']
    test_data, test_labels = m_test['X'], m_test['y']
    train_size = 73257
    x_train = train_data.reshape(-1, train_size)
    test_size = 26032
    x_test = test_data.reshape(-1, test_size)
    # 可视化图片
    # ====================
    # X = x_test.T
    # y = test_labels.flatten()
    # print(X.shape, y.shape)
    # pca = decomposition.PCA(n_components=3)
    # new_X = pca.fit_transform(X)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # =====================
    train_label = []
    test_label = []
    # 修改训练集标签
    for i in range(len(train_labels)):
        train_label.append(mapping(train_labels[i]))
    train_label = np.array(train_label)
    # 修改测试集标签
    for i in range(len(test_labels)):
        test_label.append(mapping(test_labels[i]))
        # print(mapping(test_labels[i]),"**",test_labels[i])
    test_label = np.array(test_label)
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    print("x_train", x_train.shape, "x_test", x_test.shape)
    return x_train, train_label.T, x_test, test_label.T

