#二维数据的二分类，简单实现，课堂习题

import numpy as np
import matplotlib.pyplot as plt

# 输入数据

# 线性可分
w1 = np.array([[0, 0, 1], [0, 1, 1]])
w2 = np.array([[1, 1, 1],[1,0,1]])

# 线性不可分
# w1 = np.array([[0, 0, 1], [1, 1, 1]])
# w2 = np.array([[1, 0, 1], [0, 1, 1]])
w2 = -w2
b = np.array([1, 1, 1, 1])
c = 1
flag = 0
X = np.concatenate((w1, w2), axis=0)

XX = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose())
w = np.matmul(XX, b)
e = np.matmul(X, w) - b
t = 0
iteration = 20
while 1:
    temp = min(e)
    temp1 = max(e)
    if 0 > temp > -1e-4:
        temp = 0
    if temp > 1e-3:
        deltab = e + abs(e)
        b = b + c * deltab
        w = w + c * np.matmul(XX, deltab)
        e = np.matmul(X, w) - b
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
                w = w + c * np.matmul(XX, deltab)
                e = np.matmul(X, w) - b
                t = t + 1
                if t >= iteration:
                    break
print(XX, '\n', w, '\n', e)

if flag == 0:
    # 绘图
    x1 = np.arange(-2, 4, .01)
    x2 = np.arange(-2, 4, .01)
    x1, x2 = np.meshgrid(x1, x2)
    f = w[0] * x1 + w[1] * x2 + w[2]
    plt.figure()
    plt.xlim(-0.5, 2)
    plt.ylim(-0.5, 2)
    plt.contour(x1, x2, f, 0)  # 绘制曲线
    plt.scatter(w1[:, 0], w1[:, 1], color="blue")
    plt.scatter(-w2[:, 0], -w2[:, 1], color="red")

    plt.show()
else:
    print("线性不可分")


