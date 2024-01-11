# -*-coding:utf-8 -*-

"""
# File       : template_01.py
# Time       ：2023/2/7 14:29
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 设置随机数种子保证论文可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


# 1. 参数配置
class Args:
    def __init__(self) -> None:
        self.batch_size = 1
        self.lr = 0.001
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_train = np.array(
            [-2, -1, 0, 2, 3, 4, 5, 6.0, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20]
        )
        self.data_val = np.array([15.0, 16.0, 17.0, 0.1, -3, -4])


args = Args()


# 2.网络定义
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 3.定义数据集


class Dataset_num(Dataset):
    def __init__(self, flag="train") -> None:
        self.flag = flag
        assert self.flag in ["train", "val"], "not implement"
        if self.flag == "train":
            self.data = args.data_train
        else:
            self.data = args.data_val

    def __getitem__(self, index: int):
        val = self.data[index]
        if val > 8:
            label = 1
        else:
            label = 0
        return torch.tensor(label), torch.tensor([val], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


# 4.训练循环


def train():
    train_dataset = Dataset_num(flag="train")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataset = Dataset_num("val")
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=True
    )
    model = Net(1, 32, 16, 2).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        for idx, (label, inputs) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(args.device, torch.float32)
            label = label.to(args.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()

            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print(
            "train acc = {:.3f}%, loss = {}".format(
                100 * acc / nums, np.average(train_epoch_loss)
            )
        )
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            for idx, (label, inputs) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())
                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print(
                "epoch = {}, valid acc = {:.2f}%, loss = {}".format(
                    epoch, 100 * acc / nums, np.average(val_epoch_loss)
                )
            )

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_epochs_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss, "-o", label="train_loss")
    plt.plot(valid_epochs_loss, "-o", label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "./model.pth")


def pred(val):
    model = Net(1, 32, 16, 2)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    val = torch.tensor(val).reshape(1, -1).float()
    # 需要转换成相应的输入shape，而且得带上batch_size，因此转换成shape=(1,1)这样的形状
    res = model(val)
    # real: tensor([[-5.2095, -0.9326]], grad_fn=<AddmmBackward0>) 需要找到最大值所在的列数，就是标签
    res = res.max(axis=1)[1].item()
    print(
        "predicted label is {}, {} {} 8".format(
            res, val.item(), (">" if res == 1 else "<")
        )
    )


if __name__ == "__main__":
    train()
    pred(24)
    pred(3.14)
    pred(7.8)  # 这个会预测错误，所以数据量对于深度学习很重要
