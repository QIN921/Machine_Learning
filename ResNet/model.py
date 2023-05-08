import torch.nn as nn
import torch


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()  # python2
        # super().__init__()  # python3

        # self.conv1 = conv3x3(in_channels, out_channels, stride)
        # self.bn1 = nn.BatchNorm2d(out_channels)  # 标准化来加速训练
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        out = self.features(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        # out = self.relu(out)
        out = nn.ReLU(inplace=True)
        return out


class ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        :param block: 给定残差模式 BasicBlock/BottleNeck
        resnet18和resnet34使用BasicBlock, resnet101和resnet152使用BottleNeck
        :param layers: list 每层残差块数量
        比如 Resnet34 为 [3,4,6,3]
        :param num_classes: 输出分类数
        """
        super(ResNet34, self).__init__()
        self.in_channels = 64
        # 输入为3通道照片
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 库函数中这里有replace_stride_with_dilation,使用扩张卷积来代替降采样的卷积，以达到更好的模型性能和效果
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        :param block: BasicBlock或BottleNeck
        :param out_channels:
        :param blocks: int 当前层残差模块的数目
        :param stride:
        :return:
        当stride不为1 特征图尺寸发生变化 identity需要下采样
        当残差块输入输出通道不一样时 identity需要通过1x1卷积改变通道数
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # 添加第一个残差块
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 上一层输出通道数 作为下一层输入通道数
        self.in_channels = out_channels
        # 循环添加剩余残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # 序列解包

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)
        return out
