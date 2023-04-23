import torch.nn as nn


#  定义判别器 Discriminator
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Discriminator(nn.Module):
    def __init__(self, img_area):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),  # 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),  # 进行非线性映射
            nn.Linear(512, 256),  # 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),  # 进行非线性映射
            nn.Linear(256, 1),  # 输入特征数为256，输出为1
            nn.Sigmoid(),  # sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)  # 通过鉴别器网络
        return validity  # 鉴别器返回的是一个[0, 1]间的概率


# 定义生成器 Generator
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self, img_area, latent_dim):
        super(Generator, self).__init__()

        # 模型中间块儿
        def block(in_feat, out_feat, normalize=True):  # block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]  # 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 非线性激活函数
            return layers

        # prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),  # 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),  # 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),  # 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),  # 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, img_area),  # 线性变化将输入映射 1024 to 784
            nn.Tanh()  # 将(784)的数据每一个都映射到[-1, 1]之间
        )

    # view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z, img_shape):  # 输入的是(64， 100)的噪声数据
        imgs = self.model(z)  # 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *img_shape)  # reshape成(64, 1, 28, 28)
        return imgs  # 输出为64张大小为(1, 28, 28)的图像
