import torch
import math
import numpy as np
import matplotlib.pyplot as plt


class MLP:

    def __init__(self, numInputNodes, numHiddenNodes1, numHiddenNodes2, numOutputNodes, lr, epochs):
        # 初始化
        self.numInputNodes = numInputNodes
        self.numHiddenNodes1 = numHiddenNodes1
        self.numHiddenNodes2 = numHiddenNodes2
        self.numOutputNodes = numOutputNodes
        self.lr = lr
        self.epochs = epochs

        # key_point:初始权重的设置
        self.weightInputHidden1 = np.random.normal(0.0, pow(
            self.numHiddenNodes1, -0.5), (self.numHiddenNodes1, self.numInputNodes))
        self.weightHidden1Hidden2 = np.random.normal(0.0, pow(
            self.numHiddenNodes2, -0.5), (self.numHiddenNodes2, self.numHiddenNodes1))
        self.weightHidden2Output = np.random.normal(0.0, pow(
            self.numOutputNodes, -0.5), (self.numOutputNodes, self.numHiddenNodes2))

        # 初始化所有bias均为0
        self.biasHidden1 = np.zeros((1, self.numHiddenNodes1)).T
        self.biasHidden2 = np.zeros((1, self.numHiddenNodes2)).T
        self.biasOutput = np.zeros((1, numOutputNodes)).T

    def activation_func(self, z):
        # 激活函数
        return 1.0/(1.0+math.exp((-1)*z))

    def loss_func(self, outputs, labels):
        # 交叉熵
        train_num = labels.shape[0]
        temp = np.zeros((1, train_num))
        for i in range(train_num):
            temp[0, i] = math.log(outputs[labels[i], i])
        return (-1*np.sum(temp)/train_num)

    def label_trans(self, labels):
        # 将输入的label（100*1）转化为（3*100）的新形式，便于进行后续反向传播误差处理
        train_num = labels.shape[0]
        temp = np.zeros((self.numOutputNodes, train_num))
        for i in range(train_num):
            temp[labels[i], i] = 1
        return temp

    def training(self, train_data, labels):
        # 当前神经网络训练轮数
        cur_epoch = 0
        train_num = train_data.shape[0]
        inputs = train_data.T
        loss = []

        while(cur_epoch < self.epochs):
            cur_epoch = cur_epoch + 1
            # 正向传播

            # 输入层
            hidden1_inputs = np.dot(self.weightInputHidden1, inputs)
            hidden1_outputs = np.diag(
                self.biasHidden1) * np.ones((self.numHiddenNodes1, train_num))
            hidden1_outputs = hidden1_inputs + hidden1_outputs

            # 通过激活函数处理
            for i in range(hidden1_outputs.shape[0]):
                for j in range(hidden1_outputs.shape[1]):
                    hidden1_outputs[i, j] = self.activation_func(
                        hidden1_outputs[i, j])

            # 隐层1
            hidden2_inputs = np.dot(self.weightHidden1Hidden2, hidden1_outputs)
            hidden2_outputs = np.diag(
                self.biasHidden2) * np.ones((self.numHiddenNodes2, train_num))
            hidden2_outputs = hidden2_outputs + hidden2_inputs

            # 通过激活函数处理
            for i in range(hidden2_outputs.shape[0]):
                for j in range(hidden2_outputs.shape[1]):
                    hidden2_outputs[i, j] = self.activation_func(
                        hidden2_outputs[i, j])

            # 隐层2
            output_inputs = np.dot(self.weightHidden2Output, hidden2_outputs)
            output_outputs = np.diag(self.biasOutput) * \
                np.ones((self.numOutputNodes, train_num))
            output_outputs = output_outputs + output_inputs

            for i in range(output_outputs.shape[0]):
                for j in range(output_outputs.shape[1]):
                    output_outputs[i, j] = np.exp(output_outputs[i, j])

            # 按列求和
            output_colsum = np.sum(output_outputs, axis=0)

            for i in range(output_outputs.shape[0]):
                for j in range(output_outputs.shape[1]):
                    output_outputs[i, j] = output_outputs[i,
                                                          j] / output_colsum[j]

            # 计算交叉熵
            loss.append(self.loss_func(output_outputs, labels))

            # 反向传播
            temp = np.ones((train_num, 1))

            # 计算Loss关于W3的梯度L_W3
            A3 = output_outputs
            A2 = hidden2_outputs
            Y = self.label_trans(labels)
            L_Z3 = A3 - Y
            L_W3 = np.dot(L_Z3, A2.T)
            L_b3 = np.dot(L_Z3, temp)

            # 计算Loss关于W2的梯度L_W2
            # 先计算Loss关于Z2的梯度L_Z2
            A1 = hidden1_outputs
            W3 = self.weightHidden2Output
            L_Z2 = np.multiply(np.dot(W3.T, L_Z3), np.multiply(
                A2, np.ones((A2.shape[0], A2.shape[1]))-A2))
            L_W2 = np.dot(L_Z2, A1.T)
            L_b2 = np.dot(L_Z2, temp)

            # 计算Loss关于W1的梯度L_W1
            # 同样需要先计算Loss关于Z1的梯度L_Z1
            A0 = inputs
            W2 = self.weightHidden1Hidden2
            L_Z1 = np.multiply(np.dot(W2.T, L_Z2), np.multiply(
                A1, np.ones((A1.shape[0], A1.shape[1]))-A1))
            L_W1 = np.dot(L_Z1, A0.T)
            L_b1 = np.dot(L_Z1, temp)

            # 更新权值W[1..3] 和 b[1..3]（梯度下降）
            self.weightHidden2Output -= self.lr * L_W3
            self.weightHidden1Hidden2 -= self.lr * L_W2
            self.weightInputHidden1 -= self.lr * L_W1
            self.biasOutput -= self.lr * L_b3
            self.biasHidden2 -= self.lr * L_b2
            self.biasHidden1 -= self.lr * L_b1

        return loss


if __name__ == '__main__':
    # 输入层神经元数
    input_nodes = 5
    # 隐层1神经元数
    hidden1_nodes = 4
    # 隐层2神经元数
    hidden2_nodes = 4
    # 输出层神经元数
    output_nodes = 3
    # 迭代次数
    epochs = 1000
    # 学习率
    learning_rate = 0.01

    mlp = MLP(input_nodes, hidden1_nodes, hidden2_nodes,
              output_nodes, learning_rate, epochs)
    # 生成数据
    train_data = np.random.random(size=(100, 5))
    labels = np.random.randint(0, output_nodes, size=(100, 1))
    # 训练
    LOSS_MLP = mlp.training(train_data, labels)
    # 绘制Loss与迭代次数的关系图
    plt.plot(LOSS_MLP)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()
