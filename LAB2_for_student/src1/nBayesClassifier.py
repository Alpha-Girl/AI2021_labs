import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        # 计算先验概率
        Py = {}
        yi = {}
        ySet = np.unique(trainlabel)
        for i in ySet:
            # 统计类别为i的数据量
            Py[i] = (sum(trainlabel == i)+1)/(trainlabel.shape[0]+len(ySet))
            yi[i] = sum(trainlabel == i)
        # 保存Pc: P(c) 每个类别c的概率分布
        self.Pc = Py
        ySet = yi
        print("先验概率p(c)计算完毕！")
        # 计算条件概率
        Pxy = {}

        for xIdx in range(len(featuretype)):
            # 第一层不同的属性/特征
            Xarr = traindata[:, xIdx]
            if featuretype[xIdx] == 0:
                # 离散型特征
                categoryParams = {}
                XiSet = np.unique(Xarr)
                XiSetCount = XiSet.size
                for yj, yiCount in ySet.items():
                    # 第二层是不同的分类标签
                    categoryParams[yj] = {}
                    Xiyi = Xarr[np.nonzero(trainlabel == yj)[0]]
                    for xi in XiSet:
                        # 第三层是变量X的不同值类型
                        tmp = (sum(Xiyi == xi)+1)/(Xiyi.size+XiSetCount)
                        categoryParams[yj][xi] = tmp
                Pxy[xIdx] = categoryParams
            else:
                # 连续型特征
                continuousParams = {}
                for yk, yiCount in ySet.items():
                    # 第二层是不同的分类标签
                    Xiyi = Xarr[np.nonzero(trainlabel == yk)[0]]
                    continuousParams[yk] = (Xiyi.mean(), Xiyi.std())
                Pxy[xIdx] = continuousParams
        print("条件概率p(x|c)计算完毕！")
        # 保存Pxc: P(c|x) 每个特征的条件概率
        self.Pxc = Pxy
        return

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        m, n = features.shape
        log_proba = np.zeros((m, len(self.Pc)))
        for i in range(m):
            # 第一层每个样本
            for idx, (yi, Py) in enumerate(self.Pc.items()):
                # 第二层不同类别
                # 取对数计算
                log_proba_idx = 0
                for xIdx in range(n):
                    # 第三层每种特征
                    xi = features[i, xIdx]
                    if featuretype[xIdx] == 0:
                        log_proba_idx += np.log(self.Pxc[xIdx][yi][xi])
                    else:
                        miu = self.Pxc[xIdx][yi][0]
                        sigma = self.Pxc[xIdx][yi][1]
                        t = np.exp(-(xi-miu)**2/(2*sigma**2)) / \
                            (np.power(2*np.pi, 0.5)*sigma)
                        log_proba_idx += np.log(t)
                log_proba[i, idx] = log_proba_idx+np.log(Py)
        # 选择可能性最大的
        a = np.argmax(log_proba, axis=1)
        a = a + 1
        return a.reshape(m, 1)


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label, pred)))
    print("macro-F1: "+str(get_macro_F1(test_label, pred)))
    print("micro-F1: "+str(get_micro_F1(test_label, pred)))


main()
