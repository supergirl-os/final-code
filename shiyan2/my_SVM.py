#2-2PR
#2021.4.8
#实现支持向量机算法（可使用开源代码），针对MNIST中的所有数字的识别进行评估实验。
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from sklearn.metrics import f1_score

#train_num = 10000
test_num = 10000

#数据的获取与处理
(x_train_image, y_train_label),\
(x_test_image, y_test_label) = mnist.load_data()
#处理数据维度
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
#进行归一化处理
x_train = x_Train / 255
x_test = x_Test / 255
y_train=y_train_label
y_test=y_test_label
# y_train = np_utils.to_categorical(y_train_label)
# y_test = np_utils.to_categorical(y_test_label)
train_nums=[10000,20000,30000,40000,50000,60000]
i = 0
y1 = np.zeros(6)
y2 = np.zeros(6)
for train_num in train_nums:
    # 获取一个支持向量机模型
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # 进行拟合
    predictor.fit(x_train[:train_num], y_train[:train_num])
    # 预测结果
    result = predictor.predict(x_test[:test_num])
    # 准确率估计
    accurancy = np.sum(np.equal(result, y_test[:test_num])) / test_num
    f1score = f1_score(y_test[:test_num], result, average='weighted')
    y1[i] = accurancy
    y2[i] = f1score
    print(i)
    print("train_num=",train_num)
    print(y1[i])
    print(y2[i])
    i += 1

#y1=[0.9594,0.9695,0.9742,0.9776,0.9785]
#y2=[0.9593154303218878,0.9694644919601205,]
plt.title('不同训练样本规模下的accurancy和F1_score')
plt.xlabel('训练样本规模')
plt.ylabel('accuracy/F1_score')
plt.plot(train_nums, y1, marker='o', color='red', label='accurancy')
plt.plot(train_nums, y2, marker='+', color='blue', label='F1_score')
plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
plt.grid()
plt.show()

print("The accuracy of the model is %f" %(accurancy))
print("The f1_score is", f1_score(y_test[:test_num], result, average='weighted'))