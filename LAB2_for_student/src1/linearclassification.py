from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.05,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        ''''
        需要你实现的部分
        '''

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        ''''
        需要你实现的部分
        '''



def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR=LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
