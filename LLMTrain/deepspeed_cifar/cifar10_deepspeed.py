import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed


def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # data
    # cuda
    parser.add_argument(
        "--with_cuda",
        default=False,
        action="store_true",
        help="use CPU in case there's no GPU support",
    )
    parser.add_argument(
        "--use_ema",
        default=False,
        action="store_true",
        help="whether use exponential moving average",
    )

    # train
    parser.add_argument(
        "-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30) 训练轮数",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher ",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval 每隔N个batch输出一次日志",
    )

    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)   开启混合专家模式",
    )

    parser.add_argument(
        "--ep-world-size",
        default=1,
        type=int,
        help="(moe) expert parallel world size  专家并行度",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related. MoE层列表,每个元素代表该层的专家数量",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]  两者MoE层的连接方式 串行or残差 图示https://arxiv.org/pdf/2201.05596.pdf",
    )
    parser.add_argument(
        "--top-k",
        default=1,
        type=int,
        help="(moe) gating top 1 and 2 supported  MoE每层门控选择N个专家",
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor 每个专家需要处理的最少样本数量。设置太小，可能会导致某些专家的效果不佳。设置太大，可能会导致专家之间的数据分布不均衡，影响整体性能",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter  选择专家时引入一定的随机性，以增加模型的多样性和鲁棒性",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE  是否为MoE层创建单独的参数组",
    )

    # Include DeepSpeed configuration arguments 添加deepspeed相关的配置参数
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


deepspeed.init_distributed()  # 初始化分布式后端
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 进程同步: 让rank 0下载数据集，其他rank等待
if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()

# 初始化数据集   如果不存在，则自动下载
trainset = torchvision.datasets.CIFAR10(
    root="../../data", train=True, download=True, transform=transform
)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    # rank0下载完数据后，通知其他rank开始继续执行
    torch.distributed.barrier()

# 初始化数据加载器
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="../../data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(" ".join("%5s" % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F

args = add_argument()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if args.moe:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                # create moe layers based on the number of experts
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=fc3,  # 专家的具体结构
                        num_experts=n_e,  # 专家总数
                        ep_size=args.ep_world_size,  # 专家并行度
                        use_residual=args.mlp_type
                        == "residual",  # MoE层之间是否使用残差
                        k=args.top_k,  # 每层激活的专家数
                        min_capacity=args.min_capacity,
                        noisy_gate_policy=args.noisy_gate_policy,
                    )
                )
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if args.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x


net = Net()


def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}

    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = filter(lambda p: p.requires_grad, net.parameters())  # 获取需要更新的参数
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features 初始化配置
# 1) Distributed model  转为分布式模型
# 2) Distributed data loader  传入数据集，返回分布式数据加载器
# 3) DeepSpeed optimizer 加载配置文件的优化器
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset
)

# 打印fp16是否开启
fp16 = model_engine.fp16_enabled()
print(f"fp16={fp16}")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # 定义损失函数
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(args.epochs):  # loop over the dataset multiple times 遍历N轮数据集
    running_loss = 0.0
    model_engine.train()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        # deepspeed封装后的data是一个list
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank
        )
        if fp16:
            inputs = inputs.half()  # 若开启fp16，则将输入转为half精度
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (
            args.log_interval - 1
        ):  # print every log_interval mini-batches
            print(
                "[%d, %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / args.log_interval)
            )
            running_loss = 0.0

print("Finished Training")

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.
model_engine.eval()

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
if fp16:
    images = images.half()
outputs = net(images.to(model_engine.local_rank))

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(model_engine.local_rank)).sum().item()

print(
    "Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total)
)

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(
        "Accuracy of %5s : %2d %%"
        % (classes[i], 100 * class_correct[i] / class_total[i])
    )
