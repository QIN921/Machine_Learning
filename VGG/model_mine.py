import torch.nn as nn
import torch
import math


class VGG_16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(VGG_16, self).__init__()
        self.features = nn.Sequential(
            # 2个3*3代替一个5*5
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                      padding=1),  # input[3, 224, 224] output[64, 244, 244]
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1),  # input[64, 224, 224] output[64, 244, 244]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input[64, 224, 224] output[64, 112, 112]

            # 2个3*3代替一个5*5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                      padding=1),  # input[64, 112, 112] output[128, 112, 112]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                      padding=1),  # input[128, 112, 112] output[128, 112, 112]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input[128, 112, 112] output[128, 56, 56]

            # 3个3*3代替一个7*7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                      padding=1),  # input[128, 56, 56] output[256, 56, 56]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                      padding=1),  # input[256, 56, 56] output[256, 56, 56]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                      padding=1),  # input[256, 56, 56] output[256, 56, 56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input[256, 56, 56] output[256, 28, 28]

            # 3个3*3代替一个7*7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[256, 28, 28] output[512, 28, 28]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[512, 28, 28] output[512, 28, 28]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[512, 28, 28] output[512, 28, 28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input[512, 28, 28] output[512, 14, 14]

            # 3个3*3代替一个7*7
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[512, 14, 14] output[512, 14, 14]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[512, 14, 14] output[512, 14, 14]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),  # input[512, 14, 14] output[512, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input[512, 14, 14] output[512, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # x为多个图片组成的(batch_size,channels,x,y),比如此时为(batch_size,512,7,7)
        # x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)  # 转换后有batch_size行，而-1指在不告诉函数有多少列的情况下，自动分配列数
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()
