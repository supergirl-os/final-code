# ==================================================================
# Course: Understanding Deep Neural Networks
# Teacher: Zhang Yi
# Student: Wang Yaxuan
# ID:   2019141440341
#
# Ten-category classification problem on SVHN dataset
# ====================================================================
import time
from data_loader import data_loader
from model.Network import Network


# 主函数入口
def main(model_name,L,alpha,epochs,batch_size):
    print("Start read data")
    time_1 = time.time()
    # 数据准备
    x_train, train_labels, x_test, test_labels = data_loader()

    # 模型加载

    if model_name == "Network":
        net = Network(L, alpha, epochs, batch_size)
        net.designer()
    elif model_name == "CNN":
        pass
    elif model_name == "FCN":
        pass
    else:
        print("模型名称输入有误！")
        return 0

    time_2 = time.time()
    print("read data cost ", time_2 - time_1, ' second', '\n')

    # 模型训练
    print('Start training')
    net.train(x_train, train_labels, x_test, test_labels)
    time_3 = time.time()
    print("training and predicting cost ", time_3 - time_2, ' second', '\n')

    net.show()
    net.save_model()


if __name__ == '__main__':
    model = "Network"  # 设置选择的模型种类
    L = 3            # Network 模型的层数，层数为8
    alpha = 0.01      # 学习率
    epochs = 50        # 迭代次数
    batch_size = 50    # 批次
    main(model, L, alpha, epochs, batch_size)


