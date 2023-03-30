# -*-coding:utf-8 -*-

"""
# File       : accelerate_test.py
# Time       ：2023/2/8 17:11
# Author     ：chengbo
# version    ：python 3.8
# Description：
https://mp.weixin.qq.com/s/umBHmvUhx40L-i5ucEnf5A
"""
import datetime

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def create_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root="./mnist/", train=True, download=True, transform=transform)
    # print(ds_train.data.reshape(-1, 1, 28, 28).shape)
    ds_val = torchvision.datasets.MNIST(root="./mnist/", train=False, download=True, transform=transform)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    return dl_train, dl_val


def create_net():
    net = nn.Sequential()
    net.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3))
    net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module("conv2", nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5))
    net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module("dropout", nn.Dropout2d(p=0.1))
    net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1)))
    net.add_module("flatten", nn.Flatten())
    net.add_module("linear1", nn.Linear(256, 128))
    net.add_module("relu", nn.ReLU())
    net.add_module("linear2", nn.Linear(128, 10))
    return net


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        # print("x.shape:",x.shape)
        x1 = self.conv1(x)
        # print("x1.shape:",x1.shape)
        x2 = x1.view(-1, 14 * 14 * 128)
        # print("x2.shape:",x2.shape)
        x3 = self.dense(x2)
        # print("x3.shape:",x3.shape)
        # print(x.shape, x1.shape, x2.shape, x3.shape)
        return x3


def training_loop(epochs=5, lr=1e-3, batch_size=1024, ckpt_path="checkpoint.pt", mixed_precision="no"):
    train_dataloader, eval_dataloader = create_dataloaders(batch_size)
    model = create_net()
    # model = Model()
    print(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=25 * lr, epochs=epochs,
                                                       steps_per_epoch=len(train_dataloader))
    # ======================================================================
    # initialize accelerator and auto move data/model to accelerator.device
    set_seed(42)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    accelerator.print(f"device {str(accelerator.device)} is used")
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer,
                                                                                            lr_scheduler,
                                                                                            train_dataloader,
                                                                                            eval_dataloader)
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            features, labels = batch
            preds = model(features)
            loss = nn.CrossEntropyLoss()(preds, labels)

            accelerator.backward(loss)  # loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()

        accurate = 0
        num_elems = 0
        for _, batch in enumerate(eval_dataloader):
            features, labels = batch
            with torch.no_grad():
                preds = model(features)
            predictions = preds.argmax(dim=-1)
            # gather data from multi-gpus (used when in ddp mode)
            predictions = accelerator.gather_for_metrics(predictions)
            labels = accelerator.gather_for_metrics(labels)

            accurate_preds = (predictions == labels)
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems

        accelerator.wait_for_everyone()
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        accelerator.print(f"epoch【{epoch}】@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")
        net_dict = accelerator.get_state_dict(model)
        accelerator.save(net_dict, ckpt_path + "_" + str(epoch))


if __name__ == '__main__':
    training_loop(epochs=5, lr=1e-4, batch_size=1000, ckpt_path="checkpoint.pt", mixed_precision="no")

    # x = np.random.randint(1, 10, size=(1, 1, 28, 28))
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2)(x)
    # print(y.shape)
