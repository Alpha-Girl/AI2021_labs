from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np
import math

# 实现线性回归的类


class LinearClassification:

    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.0005, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''

    def fit(self, train_features, train_labels):
        self.weight = np.array(np.random.rand(
            train_features.shape[1], train_labels.shape[1]))
        # print("w:",self.weight.shape)
        self.bias = np.zeros(train_labels.shape[1])
        x_T = train_features.transpose()
        a = x_T.dot(train_features)
        a = a + self.Lambda * \
            np.identity(train_features.shape[1], dtype=np.float32)
        a = a.transpose()
        a = np.linalg.inv(a)
        a = a.dot(x_T)
        # print("a:"+a.shape)
        for i in range(self.epochs):
            y_pred = self.predict(train_features)
            # print(y_pred.shape)
            d_w = a@(y_pred - train_labels)
            # print(d_w.shape)
            d_b = np.mean(y_pred-train_labels)
            # print(d_b.shape)
            self.weight = self.weight - self.lr * d_w
            self.bias = self.bias - self.lr * d_b

        '''
        x_T = train_features.transpose()
        a = x_T @ train_features + self.Lambda @ \
            np.identity(train_features.shape[1], dtype=np.float32)
        a = a.transpose()
        a = np.linalg.inv(a)@x_T@train_labels
        W = W - self.Lambda@a'''

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''

    def predict(self, test_features):
        y_pred_proba = test_features.dot(self.weight)+self.bias
        y_pred = y_pred_proba
        # print(y_pred_proba.shape)
        for k in range(y_pred_proba.shape[0]):

            y_pred[k][0] = math.floor(y_pred_proba[k][0])
        return y_pred


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label, pred)))
    print("macro-F1: "+str(get_macro_F1(test_label, pred)))
    print("micro-F1: "+str(get_micro_F1(test_label, pred)))


main()
