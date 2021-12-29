# 简单BP算法，前馈神经网络
import numpy as np
import utils as u
import math
import matplotlib.pyplot as plt
import pickle


class Network:
    def __init__(self, L, alpha, epochs, batch_size):
        self.L = L
        self.alpha = alpha              # initialize learning rate
        self.epochs = epochs      # training epoch
        self.batch_size = batch_size    # sample of each mini batch

        self.layer_size = []    # define number of neurons in each layer
        self.w = {}             # initialize weights
        self.a = {}
        self.z = {}
        self.delta = {}

        self.J = []             # cost of each mini batch
        self.acc = []           # accuracy of each mini batch

    def designer(self):
        # Network Architecture Design
        if self.L == 2:
            self.layer_size = [3072,
                               10]
        elif self.L == 3:
            self.layer_size = [3072,
                               256,
                               10]
        elif self.L == 5:
            self.layer_size = [3072,
                               256,
                               128,
                               64,
                               10]
        else:
            # L = 8
            self.layer_size = [3072,
                               512,
                               512,
                               128,
                               128,
                               32,
                               32,
                               10]

        # initialize weights
        for l in range(1, self.L):
            self.w[l] = 0.1 * np.random.randn(self.layer_size[l], self.layer_size[l - 1])

    def train(self, x_train, train_labels, x_test, test_labels):
        # Step 6: Train the Network
        train_size = 73257  # number of train_set
        test_size = 26032
        batch_len = math.ceil(train_size / self.batch_size)  # batch of each epoch

        for epoch in range(self.epochs):
            self.alpha = u.lr_schedule(epoch)
            index = np.random.permutation(train_size)  # for divide the training set into random batch
            for k in range(batch_len):
                start_index = k * self.batch_size
                end_index = min((k + 1) * self.batch_size, train_size)
                batch_indices = index[start_index:end_index]

                self.a[1] = x_train[:, batch_indices]   # initialize the first layer
                y = train_labels[:, batch_indices]      # get the according labels

                # forward computation
                for i in range(1, self.L):
                    self.a[i + 1], self.z[i + 1] = u.fc(self.w[i], self.a[i])

                self.delta[self.L] = (self.a[self.L] - y) * (self.a[self.L] * (1 - self.a[self.L]))
                # print("111",self.delta[self.L].shape)
                # backward computation
                for j in range(self.L - 1, 1, -1):
                    self.delta[j] = u.bc(self.w[j], self.z[j], self.delta[j + 1])
                # update weights
                for l in range(1, self.L):
                    grad_w = np.dot(self.delta[l + 1], self.a[l].T)
                    self.w[l] = self.w[l] - self.alpha * grad_w

                # self.J.append(u.cross_entropy_error(self.a[self.L], y)/ self.batch_size)
                self.J.append(u.cost(self.a[self.L], y) / self.batch_size)
                self.acc.append(u.accuracy(self.a[self.L], y))
            # Step 7: Test the Network
            self.a[1] = x_test
            y = test_labels
            # forward computation

            for l in range(1, self.L):
                self.a[l + 1], self.z[l + 1] = u.fc(self.w[l], self.a[l])
            print("y",y[1],"a", self.a[self.L][1])
            print("---------------Each epoch-----------------")
            print(epoch, "training loss:", self.J[-1], 'test loss:', u.cost(self.a[self.L], y) / test_size)
            # print(epoch, "training loss:", self.J[-1], 'test loss:', u.cross_entropy_error(self.a[self.L], y) / test_size)

    def show(self):
        # show the cost line
        plt.figure()
        plt.xlabel("Total Batch Size")  # X轴标签
        plt.ylabel("Cost")  # Y轴标签
        plt.plot(self.J)
        plt.savefig("J.png")
        plt.close()

        # show the accuracy line
        plt.figure()
        plt.xlabel("Total Batch Size")  # X轴标签
        plt.ylabel("Accuracy")  # Y轴标签
        plt.plot(self.acc)
        plt.savefig("Acc.png")
        plt.close()

    def save_model(self):
        # Step 8: Store the Network Parameters
        # save model
        model_name = 'model.pkl'
        with open(model_name, 'wb') as f:
            pickle.dump([self.w, self.layer_size], f)
        print("The model has been saved to {}".format(model_name))

