import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
import string
from os.path import join, exists
from PIL import Image

#%%

epochs = 200
batch_size = 4
num_workers = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)

#%%

class ImageDataset(Dataset):
    def __init__(self, root, origin_dir, label_dir, in_tf, out_tf):
        self.root = root
        self.origin_dir = origin_dir
        self.label_dir = label_dir
        self.in_tf = in_tf
        self.out_tf = out_tf
        self.file_list = []
        self.read_files()

    def __getitem__(self, index):

        filename = self.file_list[index]
        origin = Image.open(join(self.root, self.origin_dir, filename))
        origin = self.in_tf(origin)
        label = Image.open(join(self.root, self.label_dir, filename))
        label = self.out_tf(label)
        return tuple([filename, origin, label])

    def __len__(self):
        return len(self.file_list)

    def read_files(self):
        self.file_list=os.listdir(self.root + self.label_dir)

#%%

class TestDataset(Dataset):
    def __init__(self, root, in_tf):
        self.root = root
        self.in_tf = in_tf
        self.file_list = []
        self.read_files()

    def __getitem__(self, index):

        filename = self.file_list[index]
        origin = Image.open(join(self.root, filename))
        origin = self.in_tf(origin)
        return tuple([filename, origin])

    def __len__(self):
        return len(self.file_list)

    def read_files(self):
        self.file_list=os.listdir(self.root)

#%%

class Convert(object):
    def __init__(self, type):
        self.type = type

    def __call__(self, sample):
        
        sample = sample.convert(self.type)
        return sample

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = int(self.output_size), int(self.output_size)

        sample = sample.resize((new_h, new_w), Image.BICUBIC)

        return sample

#%%

input_size = 512
output_size = 512

train_area_root = 'data/eyelid/train_up/train_area/'
train_gland_root = 'data/eyelid/train_up/train_gland/'
val_area_root = 'data/eyelid/train_up/val_area/'
val_gland_root = 'data/eyelid/train_up/val_gland/'
test_root = 'data/eyelid/test/eye_image/'
origin_dir = 'img/'
label_dir = 'labelcol/'

in_tf = transforms.Compose([
    Convert('RGB'),
    Resize(input_size),
    transforms.ToTensor(),
])

out_tf = transforms.Compose([
    Convert('1'),
    Resize(output_size),
    transforms.ToTensor(),
])

train_area_dataset = ImageDataset(root=train_area_root, origin_dir=origin_dir,
                                  label_dir=label_dir, in_tf=in_tf, out_tf=out_tf)

val_area_dataset = ImageDataset(root=val_area_root, origin_dir=origin_dir,
                                  label_dir=label_dir, in_tf=in_tf, out_tf=out_tf)

train_gland_dataset = ImageDataset(root=train_gland_root, origin_dir=origin_dir,
                                  label_dir=label_dir, in_tf=in_tf, out_tf=out_tf)

val_gland_dataset = ImageDataset(root=val_gland_root, origin_dir=origin_dir,
                                  label_dir=label_dir, in_tf=in_tf, out_tf=out_tf)

test_dataset = TestDataset(root=test_root, in_tf=in_tf)

print('train_area_dataset: ', len(train_area_dataset))
print('val_area_dataset: ', len(val_area_dataset))
print('train_gland_dataset: ', len(train_gland_dataset))
print('val_gland_dataset: ', len(val_gland_dataset))
print('test_dataset: ', len(test_dataset))

#%%

train_area_loader = torch.utils.data.DataLoader(
    train_area_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

val_area_loader = torch.utils.data.DataLoader(
    val_area_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

train_gland_loader = torch.utils.data.DataLoader(
    train_gland_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

val_gland_loader = torch.utils.data.DataLoader(
    val_gland_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

#%%

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



"""
    构造上采样模块--左边特征提取基础模块
"""

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块
"""

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

"""
    模型主架构
"""

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        e1 = self.Conv1(x)
        # import os('e1.size=', e1.size())

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #  ('e2.size=', e2.size())

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # pid = os.getpid()('e3.size=', e3.size())

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # !kill -9 $pid('e4.size=', e4.size())

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # ('e5.size=', e5.size())

        d5 = self.Up5(e5)
        # ('d5.size=', d5.size())
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接


        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # ('d4.size=', d4.size())
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # ('d3.size=', d3.size())
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # print('d2.size=', d2.size())
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        out = nn.Sigmoid()(out)


        return out

#%%

from torchvision.utils import save_image
from sklearn.metrics import accuracy_score

model_name = './model/unet_area_up.pth'
area_save_path = './output/area/'
gland_save_path = './output/gland/'
net = U_Net().cuda()
opt = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()


# 判断是否存在模型
if os.path.exists(model_name):
    net.load_state_dict(torch.load(model_name))
    print(f"Loaded{model_name}!")
else:
    print("No Param!")

def accuracy(y_pred, y_true):
    y_pred = y_pred[0].flatten()
    y_true = y_true[0].flatten()
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    acc = 1.0 * np.sum(y_pred == y_true) / len(y_true)
    return acc

# 训练
def train(train_data, val_data):
    net.cuda()
    best_acc = 0
    less_than = 0
    for epoch in range(1, epochs + 1):
        y_pred = []
        y_true = []
        losses = []

        for filenames, inputs, labels in tqdm(train_data, desc=f"Train Epoch {epoch}/{epochs}",ncols=100):
            # print(filenames)
            inputs, labels = inputs.cuda(), labels.cuda()
            opt.zero_grad()
            out = net(inputs)
            loss = loss_func(out, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            y_pred.append(out.data.cpu().numpy())
            y_true.append(labels.cpu().numpy())
        Loss = np.mean(losses)
        Acc = accuracy(y_pred, y_true)
        print(f"Train Loss: {round(Loss, 4)}, Acc: {round(Acc, 4)} ;", end='')
        test_acc = test(val_data)
        if epoch% 10 == 0:
            output_image(val_data, os.path.join(area_save_path, f'epoch-{epoch}'))
            torch.save(net.state_dict(), model_name + f'_epoch-{epoch}')
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     less_than = 0
        # else:
        #     less_than = less_than + 1
        # if less_than > 5:
        #     break

def test(val_data):
    net.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        losses = []
        for filenames, inputs, labels in val_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            out = net(inputs)
            loss = loss_func(out, labels)
            losses.append(loss.item())
            y_pred.append(out.cpu().numpy())
            y_true.append(labels.cpu().numpy())
        Loss = np.mean(losses)
        Acc = accuracy(y_pred, y_true)
        print(f"Test Loss: {round(Loss, 4)}, Acc: {round(Acc, 4)}")
        return Acc

def pred2int(x):
    x = x[0]
    out = []
    for i in range(len(x)):
        # print(x[i])
        out.append([1 if y > 0.5 else 0 for y in x[i].data])
    out = torch.Tensor(out)
    out = torch.unsqueeze(out, 0).cuda()
    return out

def output_image(data, saved_path):
    origin_saved_path = os.path.join(saved_path, 'origin')
    pred_saved_path = os.path.join(saved_path, 'pred')
    label_saved_path = os.path.join(saved_path, 'label')

    for path in [origin_saved_path, pred_saved_path, label_saved_path]:
        os.makedirs(path,exist_ok=True)
    with torch.no_grad():
        for filenames, inputs, labels in tqdm(data, ncols=100) :
                inputs, labels = inputs.cuda(), labels.cuda()
                out = net(inputs)
                for i in range(len(inputs)):
                    x = inputs[i]
                    x_ = pred2int(out[i].cpu())
                    y = labels[i]
                    filename = filenames[i]
                    save_image(x.cpu(), os.path.join(origin_saved_path, filename))
                    save_image(x_.cpu(), os.path.join(pred_saved_path, filename))
                    save_image(y.cpu(), os.path.join(label_saved_path, filename))
        print("image save successfully !")


if __name__ == '__main__':
    train(train_area_loader, val_area_loader)
    torch.save(net.state_dict(), model_name)
