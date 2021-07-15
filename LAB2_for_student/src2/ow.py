import torch
import torch.nn as nn
from torchvision import transforms
from mlp_mixer_pytorch import MLPMixer
from torchvision.datasets import MNIST

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
    accuracy = num_correct/total
    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(
        test_loss, accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 0.001
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.manual_seed(random_seed)
    img_height = 28
    img_width = 28

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
    model = MLPMixer(
        image_size=28,
        channels=1,
        patch_size=7,
        dim=14,
        depth=3,
        num_classes=10
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mse = nn.MSELoss()

    train(model, train_loader, optimizer, n_epochs, mse)

    test(model, test_loader, mse)
