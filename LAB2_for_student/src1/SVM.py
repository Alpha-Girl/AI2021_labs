import numpy as np
import cvxopt  # 用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


# 根据指定类别main_class生成1/-1标签
def svm_label(labels, main_class):
    new_label = []
    for i in range(len(labels)):
        if labels[i] == main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)

# 实现线性回归


class SupportVectorMachine:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, kernel, C, Epsilon):
        self.kernel = kernel
        self.C = C
        self.Epsilon = Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''

    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        # d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1, x2)
        elif kernel == 'Poly':
            K = np.dot(x1, x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''

    def fit(self, train_data, train_label, test_data):
        # 构造核矩阵K，Kij = <yixi,yjxj>
        train_num = train_data.shape[0]
        K = np.zeros((train_num, train_num))
        temp = train_data
        for i in range(train_num):
            for j in range(train_num):
                K[i][j] = self.KERNEL(
                    temp[i], temp[j], self.kernel)*train_label[i]*train_label[j]

        # 构造q
        q = np.ones((train_num, 1))
        q = -1*q
        # 构造G
        G1 = np.eye(train_num, dtype=int)
        G2 = np.eye(train_num, dtype=int)
        G2 = -1*G2
        G = np.r_[G1, G2]
        # 构造h
        h1 = np.zeros((train_num, 1))
        for i in range(train_num):
            h1[i] = self.C
        h2 = np.zeros((train_num, 1))
        h = np.r_[h1, h2]
        # 构造A
        A = train_label.reshape(1, train_num)
        # 构造b
        b = np.zeros((1, 1))
        # 统一数据类型
        K = K.astype(np.double)
        q = q.astype(np.double)
        G = G.astype(np.double)
        h = h.astype(np.double)
        A = A.astype(np.double)
        b = b.astype(np.double)
        # 化成cvxopt.matrix格式
        K_1 = cvxopt.matrix(K)
        q_1 = cvxopt.matrix(q)
        G_1 = cvxopt.matrix(G)
        h_1 = cvxopt.matrix(h)
        A_1 = cvxopt.matrix(A)
        b_1 = cvxopt.matrix(b)
        # 用cvxopt（凸优化包）求解约束方程
        sol = cvxopt.solvers.qp(K_1, q_1, G_1, h_1, A_1, b_1)
        sol_x = sol['x']
        alpha = np.array(sol_x)
        indices = np.where(alpha > self.Epsilon)[0]
        bias = np.mean(
            [train_label[i] - sum([train_label[i] * alpha[i] * self.KERNEL(x, train_data[i], self.kernel) for x in train_data[indices]]) for i in indices])
        test_num = test_data.shape[0]
        # 预测
        predictions = []
        for j in range(test_num):
            prediction = bias + sum([train_label[i] * alpha[i] * self.KERNEL(
                test_data[j], train_data[i], self.kernel) for i in indices])
            predictions.append(prediction)
        prediction = np.array(predictions).reshape(test_num, 1)
        return prediction


def main():
    # 加载训练集和测试集
    Train_data, Train_label, Test_data, Test_label = load_and_process_data()
    Train_label = [label[0] for label in Train_label]
    Test_label = [label[0] for label in Test_label]
    train_data = np.array(Train_data)
    test_data = np.array(Test_data)
    test_label = np.array(Test_label).reshape(-1, 1)
    # 类别个数
    num_class = len(set(Train_label))

    # kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    # C为软间隔参数；
    # Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel = 'Gauss'
    print("Kernel:")
    print(kernel)
    C = 1
    Epsilon = 10e-5
    # 生成SVM分类器
    SVM = SupportVectorMachine(kernel, C, Epsilon)

    predictions = []
    # one-vs-all方法训练num_class个二分类器
    for k in range(1, num_class+1):
        # 将第k类样本label置为1，其余类别置为-1
        train_label = svm_label(Train_label, k)
        # 训练模型，并得到测试集上的预测结果
        prediction = SVM.fit(train_data, train_label, test_data)
        predictions.append(prediction)
    predictions = np.array(predictions)
    #one-vs-all, 最终分类结果选择最大score对应的类别
    pred = np.argmax(predictions, axis=0)+1

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label, pred)))
    print("macro-F1: "+str(get_macro_F1(test_label, pred)))
    print("micro-F1: "+str(get_micro_F1(test_label, pred)))


main()
