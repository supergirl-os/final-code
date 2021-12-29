#二维数据的二分类，简单实现
#
from functools import reduce

class VectorOp(object):
    #向量的计算
    def dot(x,y):
        #内积
        return reduce(lambda a,b:a+b,VectorOp.element_multiply(x,y),0.0)
    def element_multiply(x,y):
        return list(map(lambda x_y:x_y[0]*x_y[1],zip(x,y)))
    def element_add(x,y):
        return list(map(lambda x_y:x_y[0]+x_y[1],zip(x,y)))
    def scala_multiply(v,s):
        #每个元素和标量即常数相乘
        return map(lambda e:e*s,v)

class Perceptron(object):
    def __init__(self,input_num):
        #self.activator=activator_fun
        #self.input_vecs=input_vecs
        #self.labels=labels
        self.weights=[1,1]*input_num#参数个数
        self.bias=0.0
    def __str__(self):
        return '权重 weights\t:%s\n 偏移量 bias\t:%f\n' % (self.weights,self.bias)
    #def standard_samples(self):

    def predict(self,input_vec):
        res=VectorOp.dot(input_vec,self.weights)+self.bias
        if(res>0):
            return 1
        else:
            return 0
    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)
    def _one_iteration(self,input_vecs,labels,rate):
        samples=zip(input_vecs,labels)
        for(input_vec,label) in samples:
            output=self.predict(input_vec)
            self._update_weights(input_vec,output,label,rate)
    def _update_weights(self,input_vec,output,label,rate):
        delta=label-output
        self.weights=VectorOp.element_add(self.weights,VectorOp.scala_multiply(input_vec,rate*delta))
        self.bias+=rate*delta
        print("weights=")
        print(self.weights)
        print("bias:%f"%self.bias)
    def get_training_dataset(self):
        input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        labels = [0, 0, 1, 1]
        return input_vecs,labels

def train_and_perceptron():
    p=Perceptron(2)
       # input_vecs, labels=get_training_dataset()
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 1, 1]
    p.train(input_vecs,labels,10,1)
    return p

if __name__ == '__main__':

    perceptron_result=train_and_perceptron()
    print(perceptron_result)
    print(perceptron_result.predict([0,0]))


#训练样本分量增广化以及符号规范化：将训练样本增加一个分量1，且把来自w2的样本各分量乘以-1，得到训练模式集