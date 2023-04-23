import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import vgg
import os
import json
import time

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
                                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                 transforms.ToTensor(),  # 数据归一化到[0, 1]
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 均值, 标准差
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = os.getcwd()  # 获取当前运行目录
image_path = data_root + "/flower_data/"  # flower data set path
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()
# print(test_image[0].size(),type(test_image[0]))
# print(test_label[0],test_label[0].item(),type(test_label[0]))


# 显示图像，之前需把validate_loader中batch_size改为4
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

# 训练参数保存路径,只有这部分需要改变
model_name = "vgg16"
model_weight_path = "./vgg16.pth"
if os.path.isfile(model_weight_path):
    net = vgg(model_name=model_name, num_classes=5)
    net.load_state_dict(torch.load(model_weight_path))
else:
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)

net.to(device)
# 损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
# 优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
# 训练过程中最高准确率
best_acc = 0.0
# 记录训练loss与acc
train_loss = []
train_acc = []

# 开始进行训练和测试，训练一轮，测试一轮
for epoch in range(10):
    # train
    net.train()  # 训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # validate
    net.eval()  # 测试过程中不需要dropout，使用所有的神经元
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), model_weight_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        train_loss.append(running_loss / step)
        train_acc.append(val_accurate)

with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))

with open("./train_acc.txt", 'w') as train_ac:
    train_ac.write(str(train_acc))

print('Finished Training')
