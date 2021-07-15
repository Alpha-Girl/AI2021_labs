import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数的库

## pytorch_mlp.py

class PYTORCH_MLP(torch.nn.Module):  
    def __init__(self):
        super(PYTORCH_MLP,self).__init__()  # 
    # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(5,4) # 第一个隐含层 
        self.fc2 = torch.nn.Linear(4,4) # 第二个隐含层
        self.fc3 = torch.nn.Linear(4,3)  # 输出层
     
    def forward(self,din):
    # 前向传播， 输入值：din, 返回值 dout
        dout = torch.sigmoid(self.fc1(din))  
        dout = torch.sigmoid(self.fc2(dout))
        dout = torch.softmax(self.fc3(dout), dim=1) # 输出层使用 softmax 激活函数
        return dout