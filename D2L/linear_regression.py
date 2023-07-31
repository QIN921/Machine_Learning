import torch
from torch import nn
import random
from torch.utils import data
import d2l


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def linreg_torch_nn():
    nt = nn.Sequential(nn.Linear(2, 1))
    nt[0].weight.data.normal_(0, 0.01)
    nt[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(nt.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(nt(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(nt(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    w = nt[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = nt[0].bias.data
    print('b的估计误差：', true_b - b)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 手动实现
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    # # torch实现
    # batch_size = 10
    # data_iter = load_array((features, labels), batch_size)
    # linreg_torch_nn()
