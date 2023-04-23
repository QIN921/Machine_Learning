import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_part(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


class my_NN:
    def __init__(self):
        # 随机初始化
        self.w1 = np.random.randn(2, 2)  # 第一层网路，输入维度2，输出维度2
        self.w2 = np.random.randn(1, 2)  # 输出网络，输入维度2，输出维度1，反着写是为了便于点乘
        self.b1 = np.zeros((2, 1))
        self.b2 = 0

    def forward(self, x):
        # 可以通过这个函数去预测新的数据
        L1 = sigmoid(self.w1.dot(x.T) + self.b1)  # 得到2*1维度的输出
        out = sigmoid(self.w2.dot(L1) + self.b2)  # 得到1*1维度的输出
        return out

    def train(self, train_data, EPOCHS, lr):
        Loss = []
        for epoch in range(EPOCHS):
            temp_loss = 0
            for i, data in enumerate(train_data):
                x = np.reshape(data[:2], (2, -1))
                y = data[-1]
                # 前馈网络
                out_1 = self.w1.dot(x) + self.b1
                L1 = sigmoid(out_1)
                out = self.w2.dot(L1) + self.b2
                pred = sigmoid(out)

                # 计算损失
                loss = (y - pred) ** 2  # 现在loss是1*1矩阵
                loss = np.squeeze(loss)  # 将loss变为数字
                temp_loss += loss
                # 反向传播
                # 计算梯度
                dw_1 = (-2 * (y - pred)) * (self.w2.T * sigmoid_part(out)) * (sigmoid_part(out_1).dot(x.T))
                db_1 = (-2 * (y - pred)) * (self.w2.T * sigmoid_part(out)) * sigmoid_part(out_1)

                dw_2 = -2 * (y - pred) * sigmoid_part(out) * L1.T
                db_2 = -2 * (y - pred) * sigmoid_part(out)

                # 梯度下降
                self.w1 -= lr * dw_1
                self.w2 -= lr * dw_2
                self.b1 -= lr * db_1
                self.b2 -= lr * db_2
            temp_loss /= len(train_data)
            Loss.append(temp_loss)
            print('EPOCH:{}, loss:{:.4f}'.format(epoch, temp_loss))
        return Loss


def get_iris_data():
    iris = datasets.load_iris()
    iris_x = iris.data[:, :2]  # 只取两个特征
    iris_y = iris.target
    df = pd.DataFrame(iris_x)
    df.columns = iris.feature_names[:2]
    df['label'] = iris_y
    df = df.loc[df['label'] != 2]  # 去除第三类，只取两类
    df = df.sample(frac=1)  # 打乱数据
    return df.values


def plt_data(data: list[list[int]]):
    class_1, class_2 = [], []
    for item in data:
        if item[-1] == 0:
            class_1.append(item[:2])
        if item[-1] == 1:
            class_2.append(item[:2])
    class_1_x = [i[0] for i in class_1]
    class_1_y = [i[1] for i in class_1]
    class_2_x = [i[0] for i in class_2]
    class_2_y = [i[1] for i in class_2]
    plt.figure()
    plt.scatter(class_1_x, class_1_y)
    plt.scatter(class_2_x, class_2_y)


if __name__ == "__main__":
    nn = my_NN()
    lr = 0.01
    EPOCHS = 500
    train_data = get_iris_data()
    plt_data(train_data)

    # Loss = nn.train(train_data, EPOCHS, lr)
    # plt.figure()
    # plt.plot(np.arange(1, EPOCHS + 1, 1), Loss)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    plt.show()
