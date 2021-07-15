from MLP_manual import MLP
import numpy as np
from pytorch_mlp import PYTORCH_MLP
import torch

input_nodes = 5
hidden1_nodes = 4
hidden2_nodes = 4
output_nodes = 3
epochs = 10
learning_rate = 0.01

mlp = MLP(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate,epochs)
pytorch_mlp = PYTORCH_MLP()

train_data = np.random.random(size=(100,5))
labels = np.random.randint(0,output_nodes,size=(100,1))


# pytorch_mlp的loss衡量
lossfunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pytorch_mlp.parameters(), lr=learning_rate)
loss_dict = []
train_num = labels.shape[0]


for i in range(epochs):
    inputs = torch.from_numpy(train_data)
    inputs = inputs.to(torch.float32)
    
    labels_dup = labels.flatten()
    targets = torch.from_numpy(labels_dup)
    targets = targets.to(torch.long)
    outputs = pytorch_mlp.forward(inputs)
    loss = lossfunc(outputs,targets)
    loss_dict.append(loss)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    
# 自己实现的MLP的loss衡量
LOSS_MLP = mlp.training(train_data,labels)

for i in range(epochs):
    print("第%d轮迭代：\nloss = "%(i+1),LOSS_MLP[i])
    print("pytorch-MLP-LOSS=",loss_dict[i].item())