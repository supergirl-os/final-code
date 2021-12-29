import numpy as np
from utils import im2col, col2im,f,cross_entropy_error
from collections import OrderedDict


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # 卷积核的形状
        N, C, H, W = x.shape  # 输入数据形状
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)  # 输出数据的高
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)  # 输出数据的宽

        col = im2col(x, FH, FW, self.stride, self.pad)  # 展开的数据
        col_W = self.W.reshape(FN, -1).T  # 卷积核展开为二维数组
        out = np.dot(col, col_W) + self.b  # 计算展开后的矩阵乘积

        # 输出大小转换为合适的形状
        # transpose会更改多维数组的轴的顺序，将输出数据形状由(N,H,W,C)转变为(N,C,H,W)
        # 索引(0,1,2,3)对应着(N,H,W,C)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 更新backward过程需要用到的中间数据
        self.x = x
        self.col = col
        self.col_W = col_W

        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # 存储backward需用到的中间数据
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)  # 输出数据的高
        out_w = int(1 + (W - self.pool_w) / self.stride)  # 输出数据的宽

        # 展开输入数据
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)

        # 求出各行的最大值
        out = np.max(col, axis=1)

        # 通过reshape方法将数据转换为合适的形状
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 保存backward过程中需要用到的中间数据
        self.x = x
        self.arg_max = arg_max

        return out

    # 反向传播
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = f(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,out=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return  dx


class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out

    def backward(self,out):
        dx = np.dot(out, self.W.T)
        self.dW = np.dot(self.x.T,out)
        self.db = np.sum(out,axis=0)
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out=x.copy()
        out[self.mask] = 0
        return out

    def backward(self,out):
        out[self.mask] = 0
        dx = out
        return dx


class ConvNet:
    # """简单的ConvNet
    # conv - relu - pool - linear - relu - linear - softmax
    # input_dim是输入数据的通道，高，长
    # conv_param是卷积层的超参数
    # hidden_size是倒数第二个全连接层神经元数量
    # output_size是最后一个全连接层神经元数量
    # weight_init_std是权重的标准差
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']  # conv_param―卷积层的超参数（字典）。
        filter_size = conv_param['filter_size']  # 卷积核的大小
        filter_pad = conv_param['pad']  # 步幅
        filter_stride = conv_param['stride']  # 填充
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 初始化权重
        # 卷积层的参数初始化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # 两个Linear层的参数的初始化
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size,
                                            hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()

        # 向有序字典（OrderedDict）的layers中添加层
        # 依次命名为'Conv1'、'Relu1'、'Pool1'、'Linear1'、'Relu2'、'Affine2'
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.forward(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def backward(self, x, t):
        # 运用误差反向传播法求取梯度
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 将学习过程中计算出的权重参数梯度保存到grads字典中
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
