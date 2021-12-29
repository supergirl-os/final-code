#1-2PR
#2021.4.4
#使用sgd+keras完成实现一种多类分类器，针对MNIST中的所有数字的识别进行评估实验。

from keras.utils import np_utils
import numpy as np

np.random.seed(10)
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
#数据的获取与处理
(x_train_image, y_train_label),\
(x_test_image, y_test_label) = mnist.load_data()
#处理数据维度
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
#进行归一化处理
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

#模型训练
model = Sequential()
model.add(Dense(units = 1000,		# 定义隐藏层神经元的个数为1000
                input_dim = 784,	# 设置输入层神经元个数为784
                kernel_initializer = 'normal',	# 使用 normal distribution 正态分布的随机数来初始化weight（权重）和 bias（偏差）
                activation = 'relu')) 	# 定义激活函数relu（小于0的值为0，大于0的值不变）
model.add(Dense(units = 10,	# 定义输出层的神经元一共有10个
                kernel_initializer = 'normal',	# 使用 normal distribution 正态分布的随机数来初始化 weight 和 bias
                activation = 'softmax'))    # 定义激活函数
#不需要设置input_dim,Keras会自动按照上一层的units是256个神经元，设置这一次的input_dim是256

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy',	#设置损失函数（交叉熵损失函数）
              #optimizer = 'adam',	# 优化器使用
              optimizer='sgd',#梯度下降优化器
              metrics = ['accuracy'])
train_history = model.fit(x = x_Train_normalize, 	# 特征值
                          y = y_Train_OneHot,		# 真实值
                          validation_split = 0.2, 	# 分割比例，将60000*0.8作为训练数据，60000*0.2作为验证数据
                          epochs = 10,				# 设置训练周期
                          batch_size = 200,			# 每批训练200个数据
                          verbose = 2)				# 显示训练过程
val_loss, val_acc = model.evaluate(x_Test_normalize, y_Test_OneHot, 1)  # 评估模型对样本数据的输出结果
#classes=model.predict(x_Test_normalize)
print(val_loss)  # 模型的损失值
print(val_acc)  # 模型的准确度

tBatchSize = 128
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(x_Test_normalize, batch_size=tBatchSize, verbose=1)

# 找到每行最大的序号
result_max = np.argmax(result, axis=1)  # axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
test_max = np.argmax(y_Test_OneHot, axis=1)  # 这是结果的真实序号

result_bool = np.equal(result_max, test_max)  # 预测结果和真实结果一致的为真（按元素比较）
true_num = np.sum(result_bool)  # 正确结果的数量
print("The number of the right results is %f"%(true_num))
print("The accuracy of the model is %f" % (true_num / len(result_bool)))  # 验证结果的准确率

#进行结果的绘图
import matplotlib.pyplot as plt
from PIL import Image

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
def show_mnist(train_image,train_labels):
    n = 3
    m = 3
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n,m,i*n+j+1)
            #plt.subplots_adjust(wspace=0.2, hspace=0.8)
            index = i * n + j #当前图片的标号
            img_array = train_image[index]
            img = Image.fromarray(img_array)
            plt.title(train_labels[index])
            plt.imshow(img,cmap='Greys')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')
#show_mnist(x_Test_normalize ,y_Test_OneHot )
# accuracy 是使用训练集计算准确度
# val_accuracy 是使用验证数据集计算准确度
