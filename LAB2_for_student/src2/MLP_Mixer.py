import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST


class MLP(nn.Module):
    def __init__(self, dims, multiplxer=4):
        super(MLP, self).__init__()
        hidden = int(dims * multiplxer)

        self.out = nn.Sequential(
            nn.Linear(dims, hidden),
            nn.GELU(),
            nn.Linear(hidden, dims)
        )

    def forward(self, x):
        return self.out(x)


class MixerLayer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(MixerLayer, self).__init__()
        seq = patch_size
        dims = hidden_dim
        # LayerNorm1
        self.layer_norm1 = nn.LayerNorm(dims)
        # mlp1
        self.mlp1 = MLP(seq, multiplxer=0.5)
        # LayerNorm2
        self.layer_norm2 = nn.LayerNorm(dims)
        # mlp2
        self.mlp2 = MLP(dims)

    def forward(self, x):
        out = self.layer_norm1(x).transpose(1, 2)
        out = self.mlp1(out).transpose(1, 2)
        out += x
        out2 = self.layer_norm2(out)
        out2 = self.mlp2(out2)
        out2 += out
        return out2


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        # 图片大小
        in_dims = 28
        # 维度
        dims = hidden_dim
        # 深度
        N = depth
        # 目标类别数
        n_classes = 10
        self.embedding = nn.Linear(in_dims, dims)
        self.layers = nn.ModuleList()
        for _ in range(N):
            self.layers.append(MixerLayer(in_dims, dims))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dims, n_classes)
        self.dims = dims

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 3, 1).view(x.size(0), -1, self.dims)
        for layer in self.layers:
            out = layer(out)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out
# 定义训练函数


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_size_train = data.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pre_out = model(data)
            targ_out = torch.nn.functional.one_hot(target, num_classes=10)
            targ_out = targ_out.view((batch_size_train, 10)).float()
            loss = criterion(pre_out, targ_out)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))

# 定义测试函数


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    num_correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size_test = data.shape[0]
            data, target = data.to(device), target.to(device)
            pre_out = model(data)
            targ_out = torch.nn.functional.one_hot(target, num_classes=10)
            targ_out = targ_out.view((batch_size_test, 10)).float()
            test_loss += criterion(pre_out, targ_out)  # 将一批的损失相加
            t = pre_out.argmax(dim=1)
            num_correct += sum(t == target)
            total += batch_size_test
    # 准确率
    accuracy = num_correct/total
    # 平均损失
    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(
        test_loss, accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 0.001

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True,
                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root='./data', train=False,
                    download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #n = (28 * 28) // 4 ** 2
    model = MLPMixer(patch_size=4, hidden_dim=14, depth=12)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mse = nn.MSELoss()

    train(model, train_loader, optimizer, n_epochs, mse)

    test(model, test_loader, mse)
