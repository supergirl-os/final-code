# ==================================================================
# Course: Understanding Deep Neural Networks
# Teacher: Zhang Yi
# Student: Wang Yaxuan
# ID:   2019141440341
#
# Ten-category classification problem on SVHN dataset
# ====================================================================
import math
import numpy as np


# define the activation function
# f = lambda s: 1 / (1 + np.exp(-s))
def f(z):
    return 1.0/(1.0+np.exp(-z))


# define the derivative of activation function
df = lambda s: f(s) * (1 - f(s))


def fc(w, a):
    """
    :param w: shape [the Lth dim, the (L-1)th dim]
    :param a: shape [feature dim, batch dimension]
    :return: a_next, z_next
    """
    # % forward computing( in either vector form)
    # calculate net input
    z_next = np.dot(w, a)
    # calculate activation
    a_next = f(z_next)
    return a_next, z_next


def bc(w, z, delta_next):
    """
    :param w: shape [the Lth dim, the (L-1)th dim]
    :param z: shape [the (L-1)th dim, batch dimension]
    :param delta_next: [the Lth dim, batch dimension]
    :return: delta
    """
    # print("z_w.shape",w.shape)
    # print("z.shape",z.shape)
    # print("delta_next.shape",delta_next.shape)
    delta = np.dot(w.T, delta_next) * df(z)
    return delta


def lr_schedule(epoch):
    """
    根据epoch进行学习率的更改
    """
    learning_rate = 0.01
    if epoch>3:
        learning_rate = 0.0008
    elif epoch >5:
        learning_rate = 0.005
    elif epoch>3:
        learning_rate = 0.003
    return learning_rate


# Define Cost Function
def cost(a, y):
    """
    :param a: the predicted
    :param y: the label
    :return: the cost
    """
    J = 1/2 * np.sum((a - y)**2)
    return J


# 交叉熵loss
def cross_entropy_error(y, t):
    # 如果输入数据是一维的，即单个数据，则需要确保把y,t变为行向量而非列向量
    # 确保后面计算batch_size为1
    if y.ndim == 1:
        t = t.reshape(1, t.size) # t是one-hot标签
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] # y的行数
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# Define Evaluation Index
def accuracy(a, y):
    """
    :param a: the predicted
    :param y: the label
    :return: the accuracy
    """
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc


# 重排图像块为矩阵列
def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result


def col2im(mtx, image_size, block_size):
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = np.zeros(image_size)
    weight = np.zeros(image_size)  # weight记录每个单元格的数字重复加了多少遍
    col = 0
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[j:j + p, i:i + q] += mtx[:, col].reshape(block_size, order='F')
            weight[j:j + p, i:i + q] += np.ones(block_size)
            col += 1
    return result / weight
