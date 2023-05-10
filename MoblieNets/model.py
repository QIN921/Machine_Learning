import torch.nn as nn
import torch


class DepthSeperabelConv2d(nn.Module):
    # 可通过修改groups变为传统卷积,令groups=1,则为传统卷积
    def __init__(self, input_channels, output_channels, stride, groups=0):
        super().__init__()
        if groups != 1:
            # 深度分离卷积
            groups = input_channels
            tmp_channels = input_channels
        else:
            tmp_channels = output_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, tmp_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(tmp_channels),
            nn.ReLU6(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        if self.groups != 1:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.depthwise(x)
        return x


class MobileNet(nn.Module):
    """
        width multipler: alpha
    """

    def __init__(self, width_multiplier=1, num_classes=1000, init_weights=False):
        super().__init__()

        alpha = width_multiplier
        # 14个标准卷积，13个可分离卷积
        self.features = nn.Sequential(
            DepthSeperabelConv2d(3, int(alpha * 32), 2, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 32), int(alpha * 32), 1),
            DepthSeperabelConv2d(int(alpha * 32), int(alpha * 64), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 64), int(alpha * 64), 2),
            DepthSeperabelConv2d(int(alpha * 64), int(alpha * 128), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 1),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 2),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 256), 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 1, 1),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 2),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 2),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 1024), 1, 1),  # 标准卷积
            DepthSeperabelConv2d(int(alpha * 1024), int(alpha * 1024), 2),
            DepthSeperabelConv2d(int(alpha * 1024), int(alpha * 1024), 1, 1)  # 标准卷积
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Linear(int(alpha * 1024), num_classes),
            nn.Dropout(p=0.2),
            nn.Softmax(dim=1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
