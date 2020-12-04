### VGG

虽然最早的深度网络属于AlexNet，但是AlexNet本质上和20年前LeNet-5没有什么区别，基本上我们可以把AlexNet看成一个增大版本的LeNet。网络结构的第一次革新来自VGG网络，该网络由 Karen Simonyan 和 Andrew Zisserman 在2014年提出 （[参考链接](https://arxiv.org/pdf/1409.1556.pdf))。不同于AlexNet，VGG网络第一次成功使用小卷积（卷积核大小为3）把网络的深度推向20层；而且VGG中卷积层不改变的特征图的大小，把特征图大小的改变留给少数的池化层。这些网络的设计原则使得设计新网络变得简单，也被后来的不同网络所继承。 下图是一个分类ImageNet的VGG16的网络结构示意图（输入为224x224x3的图片）。

![png](http://cookdata.cn/media/note_images/vgg16_1537840557782_5d14.jpg)

这节将训练一个16层vgg来分类CIFAR-10数据。

```python
import time
import torch 
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans

BATCH_SIZE = 100
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10的输入图片各channel的均值和标准差
mean = [x/255 for x in [125.3,123.0,113.9]]
std = [x/255 for x in [63.0,62.1,66.7]]
n_train_samples = 50000

# 使用数据增强
train_set = dsets.CIFAR10(root='./data/cifar10',train=True,
                          transform=trans.Compose([
                              trans.RandomHorizontalFlip(),
                              trans.RandomCrop(32,padding=4),
                              trans.ToTensor(),trans.Normalize(mean,std)
                          ]),
                          download=True)
train_dl = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=6)
train_set.train_data = train_set.train_data[0:n_train_samples]
train_set.train_labels = train_set.train_labels[0:n_train_samples]

# 测试集一样
test_set = dsets.CIFAR10(root='./data/cifar10',train=False,
                         transform=trans.Compose([
                             trans.ToTensor(),trans.Normalize(mean,std)
                         ]),
                         download=True)
test_dl = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=6)

# 定义训练的辅助函数
def eval(model,criterion,dataloader):
    model.eval()
    loss ,accuracy = 0,0

    # torch.no_grad显示地告诉pytorch，前向传播地时候不需要存储计算图
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = criterion(logits,batch_y)
            loss += error.item()

            probs,pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy

def train_epoch(model,criterion,optimizer,dataloader):
    model.train()
    for batch_x,batch_y in dataloader:
        batch_x , batch_y = batch_x.to(device),batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        error = criterion(logits,batch_y)
        error.backward()
        optimizer.step()
```

#### 定义网络

- 注意这里，把卷积层和全连接层分别称为：特征提取器和分类器

- 特征提取器：
  
  - |   输入    |   3\*3卷积   |   3\*3卷积   |   pooling    |  3\*3卷积   |  3\*3卷积  |  3\*3卷积  | pooling  |
    | :-------: | :----------: | :----------: | :----------: | :---------: | :--------: | :--------: | :------: |
    | 32\*32\*3 |  32\*32\*32  |  32\*32\*32  |  16\*16\*64  | 16\*16\*64  | 16\*16\*64 | 16\*16\*64 | 8\*8\*64 |
    | 接前两格  | **3\*3卷积** | **3\*3卷积** | **3\*3卷积** | **pooling** |            |            |          |
    |           |  8\*8\*128   |  8\*8\*128   |  8\*8\*128   |  4\*4\*128  |            |            |          |
  
- 分类器
  
  - |  输入  | Linear | Linear | Linear |
    | :----: | :----: | :----: | :----: |
    | 128*16 |  1000  |  500   |   10   |

```python
def conv3x3(in_features,out_features):
    """
    该卷积操作不改变特征的大小
    """
    return nn.Conv2d(in_features,out_features,kernel_size=3,padding=1)

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.features = nn.Sequential(conv3x3(3,32),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(32),
                                      conv3x3(32,32),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(32),
                                      conv3x3(32,32),
                                      nn.ReLU(),
                                      nn.AvgPool2d(2),

                                      nn.BatchNorm2d(32),
                                      conv3x3(32,64),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(64),
                                      conv3x3(64,64),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(64),
                                      conv3x3(64,64),
                                      nn.ReLU(),
                                      nn.AvgPool2d(2),

                                      nn.BatchNorm2d(64),
                                      conv3x3(64,128),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(128),
                                      conv3x3(128,128),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(128),
                                      conv3x3(128,128),
                                      nn.ReLU(),
                                      nn.AvgPool2d(2)
                                     )
        self.classifier = nn.Sequential(nn.Linear(128*4*4,1000),
                                       nn.ReLU(),
                                       nn.Linear(1000,500),
                                       nn.ReLU(),
                                       nn.Linear(500,10)
                                    )

    def forward(self,x):
        o = self.features(x)
        o = o.view(-1,128*4*4)
        o = self.classifier(o)
        return o
    
nepochs = 50

net = VGG().to(device)
optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9,nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[40],gamma=0.1)
learn_history = []

print('开始训练VGG网络.....')
for epoch in range(nepochs):
    since = time.time()
    train_epoch(net,criterion,optimizer,train_dl)
    if (epoch)%5 == 0:
        tr_loss, tr_acc = eval(net,criterion,train_dl)
        te_loss, te_acc = eval(net,criterion,test_dl)
        learn_history.append((tr_loss,tr_acc,te_loss,te_acc))
        now = time.time()
        print('[%3d/%d, %.0f seconds] |\t tr_err: %.1e, tr_acc: %.2f\t |\t te_err: %.1e, te_acc: %.2f'%(
            epoch+1,nepochs,now-since,tr_loss,tr_acc,te_loss,te_acc))
""" 运行结果(运行太耗费资源，后面都是数据酷客上运行过的结果)：
开始训练VGG网络.....
[  1/50, 14 seconds] |     tr_err: 1.2e+00, tr_acc: 55.67  |   te_err: 1.2e+00, te_acc: 57.25
[  6/50, 15 seconds] |     tr_err: 5.2e-01, tr_acc: 82.09  |   te_err: 5.5e-01, te_acc: 81.04
[ 11/50, 15 seconds] |     tr_err: 3.5e-01, tr_acc: 87.94  |   te_err: 4.5e-01, te_acc: 85.12
[ 16/50, 15 seconds] |     tr_err: 2.8e-01, tr_acc: 90.22  |   te_err: 4.2e-01, te_acc: 86.55
[ 21/50, 15 seconds] |     tr_err: 2.0e-01, tr_acc: 93.04  |   te_err: 3.8e-01, te_acc: 88.17
[ 26/50, 15 seconds] |     tr_err: 1.7e-01, tr_acc: 94.15  |   te_err: 3.9e-01, te_acc: 88.57
[ 31/50, 15 seconds] |     tr_err: 1.4e-01, tr_acc: 95.21  |   te_err: 3.9e-01, te_acc: 88.59
[ 36/50, 15 seconds] |     tr_err: 1.2e-01, tr_acc: 96.03  |   te_err: 4.0e-01, te_acc: 88.72
[ 41/50, 15 seconds] |     tr_err: 9.8e-02, tr_acc: 96.54  |   te_err: 4.3e-01, te_acc: 88.88
[ 46/50, 15 seconds] |     tr_err: 7.5e-02, tr_acc: 97.30  |   te_err: 4.2e-01, te_acc: 89.59
"""
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([t[1] for t in learn_history],'r',label='Train')
plt.plot([t[3] for t in learn_history],'b',label='Test')
plt.ylim([75,98])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy');
```

![img](data:image/png;base64,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)

从上图可以看出，其实继续训练下去，我们的准确率还会继续提升

### ResNet

残差网络 (Residual Network)

神经网络的结构的进一步发展来自残差网络（ResNet）, 由Kaiming He于2015年提出（[参考链接](https://arxiv.org/abs/1512.03385)）。在残差网络出现之前，即使有batch normalization的帮助，训练极其深层的网络（譬如上百层）仍然是不可能的。深度神经网络难以训练的一个主要表现就是梯度消失，而为了解决这个问题Schmidhuber提出了[Highway Network](https://arxiv.org/abs/1505.00387)，在不同层直接加上一个连接，使得信号可以有效的传播到输入层,即。
$$
x^{l+1}=G^l*x^l+(1-G^l)*F(x^l)
$$
其中 $G^l(x)$ 标量：控制门的开关。 而残差网络更是将这一思想发挥到极致，相当于直接令所有的门一直开着，即
$$
x^{l+1} = x^l + F(x^l;W_t)~.
$$
因为所有的gate都是打开的，所以向后传播可以畅通无阻；非常有效的解决的梯度消失的问题。

这节训练一个残差网络来分类CIFAR-10数据。

```python
import time
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 100
mean = [x/255 for x in [125.3,123.0,113.9]]
std = [x/255 for x in [63.0,62.1,66.7]]
n_train_samples = 50000


# 使用数据增强
train_set = dsets.CIFAR10(root='./data/cifar10',train=True,
                          transform=trans.Compose([
                              trans.RandomHorizontalFlip(),
                              trans.RandomCrop(32,padding=4),
                              trans.ToTensor(),
                              trans.Normalize(mean,std)
                          ]),
                          download=True)
train_dl = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=6)

train_set.train_data = train_set.train_data[0:n_train_samples]
train_set.train_labels = train_set.train_labels[0:n_train_samples]

# 测试集一样
test_set = dsets.CIFAR10(root='./data/cifar10',train=False,
                         transform=trans.Compose([
                             trans.ToTensor(),trans.Normalize(mean,std)
                         ]),
                         download=True)
test_dl = DataLoader(test_set,batch_size=BATCH_SIZE,num_workers=6)

# 辅助函数
def eval(model,criterion,dataloader):
    model.eval()
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = criterion(logits,batch_y)
            loss += error.item()

            probs,pred_y = logits.data.max(dim=1)
            accuracy += (pred_y.data==batch_y).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy

def train_epoch(model,criterion,optimizer,dataloader):
    model.train()
    for batch_x,batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        error = criterion(logits,batch_y)
        error.backward()
        optimizer.step()
```

残差模块

该模块的结构如下图：

![VGG网络结构示意图](../figs/residual_block.png )

该模块实现了下面操作：
$$
y=x+F(x)
$$
这个操作分成两种情况：

- x 和 y 的大小以及channel完全一样
- y 和x做了对应的下采样，channel数目也发生变化 其中F是一个两层卷积神经网络操作

```python
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()
        self.stride = stride
        self.F = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride),
                               nn.BatchNorm2d(out_channels),nn.ReLU(),
                               nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                               nn.BatchNorm2d(out_channels))
        if self.stride != 1:
            self.identity = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(out_channels))
    def forward(self,x):
        if self.stride == 1:
            o = self.F(x) + x
        else:
            o = self.F(x) + self.identity(x)
        return F.relu(o)
    
rbk1 = ResidualBlock(32,32)
rbk2 = ResidualBlock(32,64,stride=2)

x = torch.rand(5,32,32,32)
y1 = rbk1(x)
y2 = rbk2(x)

print(y1.shape)
print(y2.shape)
"""
torch.Size([5, 32, 32, 32])
torch.Size([5, 64, 16, 16])
"""
```

整个网络：

|        操作        |    大小    |
| :----------------: | :--------: |
|        输入        | 32\*32\*3  |
|   conv 3\*3 , 16   | 32\*32\*16 |
| S1: (rbk, 16) \* 3 | 32\*32\*16 |
| S2: (rbk, 32) \* 3 | 16\*16\*32 |
| S3: (rbk, 64) \* 3 |  8\*8\*64  |
|   Global pooling   |  1\*1\*64  |
|       Linear       |     10     |

这里构造 14 层 ResNet：

```python
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
                                    # Stage 1
                                     ResidualBlock(16,16),ResidualBlock(16,16),ResidualBlock(16,16),
                                     # Stage 2
                                     ResidualBlock(16,32,stride=2),ResidualBlock(32,32),ResidualBlock(32,32),
                                     # Stage 3
                                     ResidualBlock(32,64,stride=2),ResidualBlock(64,64),ResidualBlock(64,64),
                                     # Global pooling
                                     nn.AvgPool2d(8)
                                    )
        self.classifier = nn.Linear(64,10)

    def forward(self,x):
        o = self.feature(x)
        o = o.view(x.size(0),-1)
        o = self.classifier(o)
        return o
    
nepochs = 50
net = ResNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),
                            lr=0.2,momentum=0.9,nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[40],gamma=0.1)
learn_history = []

print('开始训练ResNet网络.....')
for epoch in range(nepochs):
    since = time.time()
    scheduler.step()
    current_lr = scheduler.get_lr()[0]
    train_epoch(net,criterion,optimizer,train_dl)
    tr_loss, tr_acc = eval(net,criterion,train_dl)
    te_loss, te_acc = eval(net,criterion,test_dl)
    now = time.time()
    learn_history.append((tr_loss,tr_acc,te_loss,te_acc))
    print('[%3d/%d, %.0f seconds]|\t lr=%.2e,  tr_err: %.1e, tr_acc: %.2f |\t te_err: %.1e, te_acc: %.2f'%(
        epoch+1,nepochs,now-since,current_lr,tr_loss,tr_acc,te_loss,te_acc))
"""
开始训练ResNet网络.....
[  1/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.6e+00, tr_acc: 44.18 |  te_err: 1.7e+00, te_acc: 43.95
[  2/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.1e+00, tr_acc: 62.37 |  te_err: 1.1e+00, te_acc: 61.99
[  3/50, 19 seconds]|     lr=2.00e-01,  tr_err: 7.5e-01, tr_acc: 73.49 |  te_err: 8.0e-01, te_acc: 72.67
[  4/50, 19 seconds]|     lr=2.00e-01,  tr_err: 6.7e-01, tr_acc: 76.74 |  te_err: 7.0e-01, te_acc: 76.65
[  5/50, 19 seconds]|     lr=2.00e-01,  tr_err: 7.0e-01, tr_acc: 75.71 |  te_err: 7.4e-01, te_acc: 74.66
[  6/50, 19 seconds]|     lr=2.00e-01,  tr_err: 5.7e-01, tr_acc: 80.24 |  te_err: 6.7e-01, te_acc: 77.85
[  7/50, 19 seconds]|     lr=2.00e-01,  tr_err: 5.2e-01, tr_acc: 81.86 |  te_err: 5.9e-01, te_acc: 80.17
[  8/50, 19 seconds]|     lr=2.00e-01,  tr_err: 5.7e-01, tr_acc: 80.57 |  te_err: 6.6e-01, te_acc: 78.85
[  9/50, 19 seconds]|     lr=2.00e-01,  tr_err: 5.0e-01, tr_acc: 82.79 |  te_err: 5.8e-01, te_acc: 80.84
[ 10/50, 19 seconds]|     lr=2.00e-01,  tr_err: 4.1e-01, tr_acc: 85.67 |  te_err: 5.0e-01, te_acc: 83.44
[ 11/50, 19 seconds]|     lr=2.00e-01,  tr_err: 4.7e-01, tr_acc: 83.49 |  te_err: 6.0e-01, te_acc: 81.21
[ 12/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.8e-01, tr_acc: 86.75 |  te_err: 4.8e-01, te_acc: 84.27
[ 13/50, 20 seconds]|     lr=2.00e-01,  tr_err: 3.7e-01, tr_acc: 87.01 |  te_err: 4.9e-01, te_acc: 84.25
[ 14/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.6e-01, tr_acc: 87.65 |  te_err: 4.7e-01, te_acc: 84.76
[ 15/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.4e-01, tr_acc: 88.17 |  te_err: 4.8e-01, te_acc: 84.16
[ 16/50, 18 seconds]|     lr=2.00e-01,  tr_err: 3.4e-01, tr_acc: 88.16 |  te_err: 4.5e-01, te_acc: 85.40
[ 17/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.2e-01, tr_acc: 88.79 |  te_err: 4.4e-01, te_acc: 85.88
[ 18/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.0e-01, tr_acc: 89.32 |  te_err: 4.3e-01, te_acc: 86.18
[ 19/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.9e-01, tr_acc: 89.89 |  te_err: 4.4e-01, te_acc: 85.79
[ 20/50, 18 seconds]|     lr=2.00e-01,  tr_err: 3.5e-01, tr_acc: 88.11 |  te_err: 5.2e-01, te_acc: 83.93
[ 21/50, 18 seconds]|     lr=2.00e-01,  tr_err: 3.1e-01, tr_acc: 89.00 |  te_err: 4.9e-01, te_acc: 84.72
[ 22/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.7e-01, tr_acc: 90.42 |  te_err: 4.3e-01, te_acc: 86.03
[ 23/50, 19 seconds]|     lr=2.00e-01,  tr_err: 3.0e-01, tr_acc: 89.51 |  te_err: 4.8e-01, te_acc: 85.21
[ 24/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.4e-01, tr_acc: 91.76 |  te_err: 4.1e-01, te_acc: 87.21
[ 25/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.6e-01, tr_acc: 90.87 |  te_err: 4.2e-01, te_acc: 87.04
[ 26/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.5e-01, tr_acc: 91.23 |  te_err: 4.3e-01, te_acc: 86.74
[ 27/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.2e-01, tr_acc: 92.11 |  te_err: 4.1e-01, te_acc: 87.17
[ 28/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.3e-01, tr_acc: 91.98 |  te_err: 4.0e-01, te_acc: 87.58
[ 29/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.2e-01, tr_acc: 92.38 |  te_err: 4.1e-01, te_acc: 87.53
[ 30/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.0e-01, tr_acc: 92.86 |  te_err: 4.1e-01, te_acc: 87.41
[ 31/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.0e-01, tr_acc: 92.73 |  te_err: 4.2e-01, te_acc: 87.48
[ 32/50, 18 seconds]|     lr=2.00e-01,  tr_err: 1.8e-01, tr_acc: 93.86 |  te_err: 3.9e-01, te_acc: 88.28
[ 33/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.8e-01, tr_acc: 93.57 |  te_err: 3.8e-01, te_acc: 88.04
[ 34/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.9e-01, tr_acc: 93.28 |  te_err: 4.1e-01, te_acc: 87.97
[ 35/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.9e-01, tr_acc: 93.43 |  te_err: 4.0e-01, te_acc: 87.99
[ 36/50, 18 seconds]|     lr=2.00e-01,  tr_err: 1.8e-01, tr_acc: 93.69 |  te_err: 4.1e-01, te_acc: 87.95
[ 37/50, 19 seconds]|     lr=2.00e-01,  tr_err: 2.1e-01, tr_acc: 92.47 |  te_err: 5.0e-01, te_acc: 86.63
[ 38/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.9e-01, tr_acc: 93.44 |  te_err: 4.5e-01, te_acc: 87.20
[ 39/50, 18 seconds]|     lr=2.00e-01,  tr_err: 2.0e-01, tr_acc: 92.86 |  te_err: 4.5e-01, te_acc: 86.93
[ 40/50, 19 seconds]|     lr=2.00e-01,  tr_err: 1.7e-01, tr_acc: 93.91 |  te_err: 4.2e-01, te_acc: 87.65
[ 41/50, 19 seconds]|     lr=2.00e-02,  tr_err: 1.0e-01, tr_acc: 96.39 |  te_err: 3.5e-01, te_acc: 89.81
[ 42/50, 19 seconds]|     lr=2.00e-02,  tr_err: 9.4e-02, tr_acc: 96.77 |  te_err: 3.5e-01, te_acc: 90.14
[ 43/50, 19 seconds]|     lr=2.00e-02,  tr_err: 8.9e-02, tr_acc: 97.01 |  te_err: 3.5e-01, te_acc: 89.97
[ 44/50, 19 seconds]|     lr=2.00e-02,  tr_err: 8.4e-02, tr_acc: 97.06 |  te_err: 3.6e-01, te_acc: 90.21
[ 45/50, 19 seconds]|     lr=2.00e-02,  tr_err: 8.1e-02, tr_acc: 97.27 |  te_err: 3.6e-01, te_acc: 90.21
[ 46/50, 19 seconds]|     lr=2.00e-02,  tr_err: 7.9e-02, tr_acc: 97.28 |  te_err: 3.6e-01, te_acc: 90.15
[ 47/50, 19 seconds]|     lr=2.00e-02,  tr_err: 7.7e-02, tr_acc: 97.28 |  te_err: 3.7e-01, te_acc: 90.08
[ 48/50, 19 seconds]|     lr=2.00e-02,  tr_err: 7.5e-02, tr_acc: 97.44 |  te_err: 3.7e-01, te_acc: 90.24
[ 49/50, 19 seconds]|     lr=2.00e-02,  tr_err: 7.1e-02, tr_acc: 97.63 |  te_err: 3.7e-01, te_acc: 90.22
[ 50/50, 19 seconds]|     lr=2.00e-02,  tr_err: 7.0e-02, tr_acc: 97.65 |  te_err: 3.7e-01, te_acc: 90.23
"""
```

```python
# 实现 bottleneck 结构
class BottleneckBlock(nn.Module):
    def __init__(self, inplanes,outplanes,stride=1):
        super(BottleneckBlock, self).__init__()     
        self.F = nn.Sequential(
                       nn.Conv2d(inplanes, inplanes, kernel_size=1),
                       nn.BatchNorm2d(inplanes),
                       nn.ReLU(True),
                       conv3x3(inplanes,inplanes,stride=stride),
                       nn.BatchNorm2d(inplanes),
                       nn.ReLU(True),
                       nn.Conv2d(inplanes,outplanes,kernel_size=1),
                       nn.BatchNorm2d(outplanes)
                    )
        if stride==1 and (inplanes == outplanes):
            self.short_cut = ShortCut(inplanes,outplanes,'A')
        else:
            self.short_cut = ShortCut(inplanes,outplanes,'B')

    def forward(self, x):
        o = self.F(x) + self.short_cut(x)
        o = F.relu(o)
        return o
    
rbk1 = BottleneckBlock(32,32)
rbk2 = BottleneckBlock(32,64,stride=2)
x = Variable(torch.rand(10,32,16,16))
y1 = rbk1(x)
y2 = rbk2(x)

print('input shape: \t\t', x.shape)
print('non-downsample case:\t',y1.shape)
print('downsample case:\t',y2.shape)
"""
input shape:          torch.Size([10, 32, 16, 16])
non-downsample case:     torch.Size([10, 32, 16, 16])
downsample case:     torch.Size([10, 64, 8, 8])
"""
```

### GAN

对抗生成网络（Generative Adversarial Networks, GAN）是现在最流行的一种生成模型，该模型由Goodfellow在2014年提出（[参考链接](https://arxiv.org/abs/1406.2661)）

假设有数据 $\{x_1,x_2,\dots,x_n\}$ ，由分布 $\pi (x)$ 独立抽样而来，而目的是建立一个生成模型：
$$
\overset{\sim}\pi (x) = \int \delta (x-G(z))Q(z)dz 
$$
使得 $\overset{\sim}\pi (\cdot)$ 的分布和 $\pi (\cdot)$ 越接近越好。这其中 $Q(\cdot)$ 一般是低纬简单分布，譬如高斯和均匀分布；而 $G(z)$ 是生成器。为了学习生成器 $G$ ，目标就变成是使得：
$$
\overset{\sim}\pi(\cdot)~\approx~\pi_N(\cdot)~:=~\frac{1}{N}\sum\limits_{i=1}^N\delta(\cdot -x_i)~.
$$
这里的核心是如何度量经验分布和我们的模型之间的差别。对抗生成网络的做法是学习一个分类器去区分它们：如果这两组数据分不开，那么学到的生成器 $G$ 就越好。假设分类器记成 $D:\mathbb{R}^d \to [0,1]$ 表示该样本是来自真实数据的概率，目标函数就可以写成下面的极值问题：
$$
\min\limits_{G}\max\limits_{D}\mathbb{E}_{x\sim\pi_n}[\log(D(x))]+\mathbb{E}_{z\sim Q(\cdot)}[\log(1-D(G(z)))]~.
$$

```python
import itertools
import math
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.datasets as dsets
import torchvision.transforms as trans

%matplotlib inline
import matplotlib.pyplot as plt

BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 我们将每个像素值scale到[-1,1]
transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载MNIST数据
train_set = dsets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, 
                                         shuffle=True,num_workers=6)

# 定义判别器
	# 784 -> 256 -> 256 -> [0,1]
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.F = nn.Sequential(nn.Linear(784,256),nn.ReLU(),
                              nn.Linear(256,256),nn.ReLU(),
                              nn.Linear(256,1),nn.Sigmoid())
    def forward(self,x):
        o = x.view(x.size(0),-1)
        o = self.F(o)
        return o
    
# 定义生成器
	# z -> 256 -> 256 -> 784
class Generator(nn.Module):
    def __init__(self,z_dim=20):
        super(Generator,self).__init__()
        self.F = nn.Sequential(nn.Linear(z_dim,256),nn.ReLU(),
                              nn.Linear(256,256),nn.ReLU(),
                              nn.Linear(256,784),nn.Tanh())
    def forward(self,z):
        o = self.F(z)
        o = o.view(z.size(0),28,28)
        return o
    
# 准备相关变量
z_dim = 20
netD = Discriminator().to(device)
netG = Generator(z_dim).to(device)
optimizerD = torch.optim.Adam(netD.parameters(),lr=1e-4)
optimizerG = torch.optim.Adam(netG.parameters(),lr=1e-4)
nepochs = 200
criterion = nn.BCELoss()
one = torch.ones(BATCH_SIZE,1).to(device)
zero = torch.zeros(BATCH_SIZE,1).to(device)

# 训练
accR_L = []
accF_L = []
real_labels = torch.ones(BATCH_SIZE,1).to(device)
fake_labels = torch.zeros(BATCH_SIZE,1).to(device)
noise_z = torch.randn(BATCH_SIZE,z_dim).to(device)

for epoch in range(nepochs):
    since = time.time()
    for batch_x,_ in dataloader:
        real_x = batch_x.to(device)

        # 更新判别器
        optimizerD.zero_grad()
        real_y = netD(real_x)
        z = noise_z.normal_(0,1)
        fake_x = netG(z)
        fake_y = netD(fake_x.detach())
        
        errD = criterion(real_y,real_labels) + criterion(fake_y,fake_labels)
        errD.backward()
        optimizerD.step()
        accR_L.append(real_y.data.mean())
        accF_L.append(fake_y.data.mean())

        # 更新生成器
        optimizerG.zero_grad()
        z = noise_z.normal_(0,1)
        fake_x = netG(z)
        fake_y = netD(fake_x)
        
        errG = criterion(fake_y,real_labels)
        errG.backward()
        optimizerG.step()

    now = time.time()
    print('[%d/%d, %.0f seconds]|\t err_D: %.4f \t err_G: %.4f'%(
        epoch, nepochs,now-since,errD,errG))
#     vutils.save_image(fake_x.data.cpu().view(-1,1,28,28),
#                       'save_gan/fake%d.png'%(epoch+1),
#                        normalize=True,nrow=10)
"""
[0/200, 6 seconds]|     err_D: 0.2367   err_G: 2.7937
[1/200, 5 seconds]|     err_D: 0.4675   err_G: 2.8868
[2/200, 5 seconds]|     err_D: 0.5432   err_G: 2.0266
.
.
.
[197/200, 6 seconds]|     err_D: 0.8234   err_G: 1.3695
[198/200, 5 seconds]|     err_D: 0.8276   err_G: 1.7026
[199/200, 5 seconds]|     err_D: 0.7604   err_G: 1.9100
"""
```

```python
# 检查生成的图片
noise_z = torch.randn(25,z_dim).to(device)
fake_x = netG(noise_z).data.cpu().view(25,1,28,28)
img = vutils.make_grid(fake_x,nrow=5,normalize=True)
img = img.numpy().transpose([1,2,0])
plt.imshow(img)
```

![img](data:image/png;base64,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%0A4XA4QvkvIgmYQP0Mjw0AAAAASUVORK5CYII=)

```python
# 可视化迭代过程
import numpy as np

# Smooth curve
def smooth_data(x,ksize=5):
    kernel = np.ones(ksize)/ksize
    x_smooth = np.convolve(x,kernel,mode='valid')
    return x_smooth

accR_L_smooth = smooth_data(accR_L,120)
accF_L_smooth = smooth_data(accF_L,120)

plt.plot(accR_L_smooth,'-b',label='accuracy on real images')
plt.plot(accF_L_smooth,'-r',label='accuracy on fake images')
plt.plot(accR_L,'-b',alpha=0.3)
plt.plot(accF_L,'-r',alpha=0.3)
plt.legend()
```

![img](data:image/png;base64,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)

由上图可以看出，

- GAN的训练过程及其不稳定。
- 根据GAN原始文章中的分析，GAN的最优解应该在真假样本上准确率都是50%的时候取到，即真假样本无法辨别。但是根据 上图的趋势，即使继续训练下去，也无法达到这个最优解，因为判别器在原始样本上的准确率要远远高于假样本。一个可能的解释是判别器过于强大，导致它总是能把假样本给区分出来。

等以后有空的时候尝试不同的超参数，看对收敛性的影响。包括

- 生成器和判别器换优化算法
- 尝试学习率（5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6）看能否成功训练

### VAE

变分自编码器

```python
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as trans

%matplotlib inline
import matplotlib.pyplot as plt

BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载MNIST数据
train_set = dsets.MNIST(root='./data/mnist', train=True, download=True,transform=trans.ToTensor())
train_dl = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=6)
```

定义自编码器

|  模型  |       网络结构       |
| :----: | :------------------: |
| 编码器 | $784-1000-1000-2z_d$ |
| 解码器 | $z_d -1000-1000-784$ |

```python
class VAE(nn.Module):
    def __init__(self,z_dim):
        super(VAE,self).__init__()
        W = 1000
        self.z_dim = z_dim
        self.encoder = nn.Sequential(nn.Linear(784,W),
                                    nn.ReLU(),nn.Linear(W,W),nn.ReLU(),
                                    nn.Linear(W,2*z_dim))
        self.decoder = nn.Sequential(nn.Linear(z_dim,W),nn.ReLU(),
                                    nn.Linear(W,W),nn.ReLU(),
                                    nn.Linear(W,784),nn.Sigmoid())
    def forward(self,x):
        o = self.encoder(x)
        mu,log_var = o[:,0:self.z_dim], o[:,self.z_dim:]
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_var*0.5)
        x_rec = self.decoder(z)
        return x_rec,mu,log_var
```

```python
# 实现训练代码
def train_epoch(model,optimizer,dataloader):
    loss_avg = 0
    for x,_ in dataloader:
        x = x.view(-1,784).to(device)
        optimizer.zero_grad()
        x_rec, mu, log_var = model(x)
        loss_rec = F.mse_loss(x_rec,x,reduction='sum')
        loss_rec = F.B
        kl_d = 0.5*(mu.pow(2)+torch.exp(log_var) - log_var - 1).sum()
        loss_tot = (loss_rec+kl_d)/x.size(0)
        loss_tot.backward()
        optimizer.step()
        loss_avg += loss_tot
    return loss_avg/len(dataloader)

# 实例化模型和优化器
Z_dim = 40
model = VAE(Z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
nepochs = 40

# 训练
for epoch in range(nepochs):
    loss = train_epoch(model,optimizer,train_dl)
    print('%d/%d, %.2e'%(epoch+1,nepochs,loss))
"""
1/40, 4.78e+01
2/40, 3.75e+01
3/40, 3.40e+01
.
.
.
38/40, 2.73e+01
39/40, 2.73e+01
40/40, 2.72e+01
"""
```

```python
# 查看生成器效果
z = torch.randn(100,Z_dim).to(device)
x = model.decoder(z)
x = x.data.view(-1,1,28,28).cpu()

import matplotlib.pyplot as plt
import torchvision
X = torchvision.utils.make_grid(x,nrow=10)
X = X.permute([1,2,0]).numpy()
plt.figure(figsize=(8,8))
plt.imshow(X);
```

![](images/效果.JPG)

Latent 空间差值

让 $z$ 是latent空间的变量。$z(0)$ 和 $z(1)$ 分别对应于两张图片；那么我们关心的是线段：
$$
z(t)=(1−t)z(0)+tz(1)
$$
 的隐变量对应的图片是什么样子的。

- 取两张图片投影到 latent 空间

```python
X = train_set.train_data[0:2]
X = X.unsqueeze(1).float()/255
X = X.view(-1,784).to(device)

e = model.encoder(X)
mu,log_var = e[:,0:Z_dim],e[:,Z_dim:]
```

* 线性差值

```python
t_s = torch.linspace(0,1,10)
x_s = []
for t in t_s:
    t = t.item()
    mu_cur = mu[0] * t + mu[1] * (1-t)
    log_var_cur = log_var[0] * t + log_var[1] * (1-t)

    eps = torch.rand_like(mu_cur)
    z = mu_cur + eps * torch.exp(log_var_cur/2)
    x_cur = model.decoder(z)

    x_s.append(x_cur)
```

* 可视化，线段对应的图片

```python
plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    x = x_s[i].cpu().view(28,28).detach().numpy()
    plt.imshow(x);
    plt.axis('off')
```

![](images/05.JPG)

可以看出0和5的变化，是通过闭合5的两个开口。

## 预训练与迁移学习

### 预训练模型

PyTorch提供了大量的在ImageNet上预训练好的模型，为我们实际使用带来了巨大方便。一方面，便于我们在这些标准模型上进行探索和研究；另一方面，在解决实际问题的时候，我们可以使用其提供的特征（feature）来帮助解决自己的问题。深度学习区别于浅层学习的核心就在于，深度学习能学习到丰富的特征（feature）。

Pytorch的预训练模型，实现在torchvison.models里面([链接](https://pytorch.org/docs/0.3.0/torchvision/models.html?highlight=resnet18#torchvision.models.resnet18))。所有的网络以及其对应的single-crop的误差见下表

|             Network             | Top-1 error | Top-5 error |
| :-----------------------------: | :---------: | :---------: |
|             AlexNet             |    43.45    |    20.91    |
|             VGG-11              |    30.98    |    11.37    |
|             VGG-13              |    30.07    |    10.75    |
|             VGG-16              |    28.41    |    9.62     |
|             VGG-19              |    27.62    |    9.12     |
| VGG-11 with batch normalization |    29.62    |    10.19    |
| VGG-13 with batch normalization |    28.45    |    9.63     |
| VGG-16 with batch normalization |    26.63    |    8.50     |
| VGG-19 with batch normalization |    25.76    |    8.15     |
|            ResNet-18            |    30.24    |    10.92    |
|            ResNet-34            |    26.70    |    8.58     |
|            ResNet-50            |    23.85    |    7.13     |
|           ResNet-101            |    22.63    |    6.44     |
|           ResNet-152            |    21.69    |    5.94     |
|         SqueezeNet 1.0          |    41.90    |    19.58    |
|         SqueezeNet 1.1          |    41.81    |    19.38    |
|          Densenet-121           |    25.35    |    7.83     |
|          Densenet-169           |    24.00    |    7.00     |
|          Densenet-201           |    22.80    |    6.43     |
|          Densenet-161           |    22.35    |    6.20     |
|          Inception v3           |    22.55    |    6.44     |

这节使用预训练网络，并用它来构造对抗性样本。

```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as trans
```

下面代码，加载一个预先在ImageNet上训练好的34层残差网络

- pretrained=True 表示网络参数是在ImageNet上预训练的
- pretrained=False 表示网络参数是随机初始化的

```python
model = models.resnet34(pretrained=True)
```

下面用这个网络去分类一个图片

读入图片

- 注意：PyTorch的预训练模型，除了Inception v3都假设输入的大小为 224*224

```python
from PIL import Image
img = Image.open('./figs/green-peper.jpg').resize((224,224))
img_np = np.asarray(img).astype(np.float32)/255
plt.imshow(img_np); plt.axis('off')

# 使用transform提供的函数将Image转化成tensor
img_th = trans.ToTensor()(img)
```

![img](data:image/png;base64,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)

图片预处理

所有预训练的模型的输入图片，每个像素的值必须在 0 到 1之间，再用 $mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225]$ 做归一化。下面把这个归一化过程抽象成一个Layer，方便后面使用

```python
# 定义归一化接口
class Normalize(nn.Module):
    def __init__(self,device=torch.device('cpu')):
        super(Normalize,self).__init__()
        self.mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

    def forward(self,x):
        if x.ndimension() ==3:
            x = x.unsqueeze(0) # 3x224x224 ==> 1x3x224x224
        o = (x-self.mean)/self.std
        return o
```

分类图片

- 如果模型使用了batch normalization 或者 dropout，我们要区别训练和推断的不同。

```python
# 下面这个网络，不需要再考虑输入的归一化
net = nn.Sequential(Normalize(),model)
net.eval()

# 网络的直接输出并不是概率，而是logit
logit = net(img_th)
probs = F.softmax(logit,dim=1).squeeze().data.numpy()
```

- 可视化预测概率

```python
plt.plot(probs,'-o')

idx = np.argmax(probs)
print('predicted label is %d'%(idx))
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAD8CAYAAACMwORRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAEB9JREFUeJzt3X+s3Xddx/Hna+02JiBj9Eq2ttoRy7Qh6MjNGMHEBQb7%0AEbJiMLpG48CFxsgQhMxswQydf2ENiMlEpiJKdGPMpTazWnXMmBA3d0fnfnQULuPH2oG7wDYTqWyF%0At3+cb+fZXdd7vt25Pfd8+nwkJz3fz/dzz/l876d53e/9fD7nflJVSJLacsKkGyBJGj/DXZIaZLhL%0AUoMMd0lqkOEuSQ0y3CWpQYa7JDXIcJekBhnuktSg1ZN64zVr1tSGDRsm9faSNJXuvvvub1XVzFL1%0Algz3JJ8A3gI8WlWvOsz5AB8FLga+C7y9qj6/1Otu2LCBubm5papJkoYk+doo9UYZlvkkcOERzl8E%0AbOweW4GPjfLGkqTls2S4V9W/Ad85QpXNwF/VwB3AqUlOH1cDJUn9jWNCdS3w8NDxvq5MkjQhx3S1%0ATJKtSeaSzC0sLBzLt5ak48o4wn0/sH7oeF1X9ixVdX1VzVbV7MzMkpO9kqSjNI6lkDuAK5LcCLwW%0AeKKqvjGG15WkFWP77v1s27WXRx4/wBmnnsKVF5zFW89euSPQoyyFvAE4D1iTZB/wQeBEgKr6E2An%0Ag2WQ8wyWQr5juRorSZOwffd+rr7lPg489X0A9j9+gKtvuQ9gxQb8kuFeVVuWOF/Au8bWIklaYbbt%0A2vt0sB9y4Knvs23X3hUb7v75AUlawiOPH+hVvhIY7pK0hDNOPaVX+UpguEvSEq684CxOOXHVM8pO%0AOXEVV15w1oRatLSJ/eEwSZoWh8bV3/vpewBYOwWrZbxzl6QRDAf55656w4oOdjDcJalJhrskNchw%0Al6QGGe6S1CDDXZIaZLhLUoMMd0lqkOEuSQ0y3CWpQYa7JDXIcJekBhnuktQgw12SGmS4S1KDDHdJ%0AapDhLkkNMtwlqUGGuyQ1yHCXpAYZ7pLUIMNdkhpkuEtSgwx3SWqQ4S5JDTLcJalBhrskNchwl6QG%0AjRTuSS5MsjfJfJKrDnP+R5PcnmR3knuTXDz+pkqSRrVkuCdZBVwHXARsArYk2bSo2m8DN1XV2cCl%0AwB+Pu6GStFJU1aSbsKRR7tzPAear6qGqehK4Edi8qE4BP9w9fwnwyPiaKEnqa/UIddYCDw8d7wNe%0Au6jO7wD/lOTdwAuB88fSOklagaogmXQrjmxcE6pbgE9W1TrgYuBTSZ712km2JplLMrewsDCmt5Yk%0ALTZKuO8H1g8dr+vKhl0O3ARQVf8OvABYs/iFqur6qpqtqtmZmZmja7EkaUmjhPtdwMYkZyY5icGE%0A6Y5Fdb4OvBEgyU8yCHdvzSU1aeVPp44Q7lV1ELgC2AU8yGBVzANJrk1ySVft/cA7k/wncAPw9pqG%0A6WRJatQoE6pU1U5g56Kya4ae7wFeP96mSdLKNLh3Xdkzqn5CVZIaZLhLUoMMd0nqaRomFA13SWqQ%0A4S5JPU3DWkDDXZIaZLhLUoMMd0nqqaZgStVwl6QGGe6S1JMTqpKkiTDcJalBhrskNchwl6QGGe6S%0A1JMTqpKkiTDcJalBhrsk9eQnVCVJE2G4S1JPTqhKkibCcJekBhnuktTTFIzKGO6S1CLDXZJ6qimY%0AUTXcJalBhrskNchwl6SeVv6gjOEuSU0y3CWppymYTzXcJalFhrskNWikcE9yYZK9SeaTXPUcdX4h%0AyZ4kDyT5m/E2U5JWkCkYllm9VIUkq4DrgDcB+4C7kuyoqj1DdTYCVwOvr6rHkvzIcjVYkrS0Ue7c%0AzwHmq+qhqnoSuBHYvKjOO4HrquoxgKp6dLzNlKSVo5XNOtYCDw8d7+vKhr0SeGWSzyW5I8mFh3uh%0AJFuTzCWZW1hYOLoWS5KWNK4J1dXARuA8YAvwp0lOXVypqq6vqtmqmp2ZmRnTW0uSFhsl3PcD64eO%0A13Vlw/YBO6rqqar6CvBFBmEvSc1pZZ37XcDGJGcmOQm4FNixqM52BnftJFnDYJjmoTG2U5LUw5Lh%0AXlUHgSuAXcCDwE1V9UCSa5Nc0lXbBXw7yR7gduDKqvr2cjVaknRkSy6FBKiqncDORWXXDD0v4H3d%0AQ5KaNgWjMn5CVZJaZLhLUk/uxCRJmgjDXZIaZLhLUk8rf1DGcJekJhnuktTTFMynGu6S1CLDXZIa%0AZLhLUk+t/D13SdKUMdwlqa+Vf+NuuEtSiwx3SWqQ4S5JPU3BqIzhLkktMtwlqSc/oSpJmgjDXZIa%0AZLhLUk9+QlWSNBGGuyT15ISqJGkiDHdJapDhLkk9TcGojOEuSS0y3CWpp5qCGVXDXZIaZLhLUoMM%0Ad0nqaQpGZQx3SWrRSOGe5MIke5PMJ7nqCPXelqSSzI6viZKkvpYM9ySrgOuAi4BNwJYkmw5T78XA%0Ae4A7x91ISVI/o9y5nwPMV9VDVfUkcCOw+TD1fg/4EPC/Y2yfJOkojBLua4GHh473dWVPS/IaYH1V%0A/f0Y2yZJK9JxMaGa5ATgw8D7R6i7NclckrmFhYXn+9aSpOcwSrjvB9YPHa/ryg55MfAq4F+TfBU4%0AF9hxuEnVqrq+qmaranZmZuboWy1JE9TKZh13ARuTnJnkJOBSYMehk1X1RFWtqaoNVbUBuAO4pKrm%0AlqXFkqQlLRnuVXUQuALYBTwI3FRVDyS5Nskly91ASVJ/q0epVFU7gZ2Lyq55jrrnPf9mSdLKdVxM%0AqEqSVh7DXZJ6moIbd8NdklpkuEtSgwx3SerJnZgkSRNhuEtSgwx3Sepp5Q/KGO6S1CTDXZJ6moL5%0AVMNdklpkuEtSgwx3Sept5Y/LGO6S1CDDXZJ6ckJVkjQRhrskNchwl6SepmBUxnCXpBYZ7pLUkxOq%0AkqSJMNwlqUGGuyT1VFMwpWq4S1KDDHdJ6skJVUnSRBjuktQgw12SenJYRpI0EYa7JPXkUkhJ0kQY%0A7pLUoJHCPcmFSfYmmU9y1WHOvy/JniT3JrktyY+Nv6mStDI0MaGaZBVwHXARsAnYkmTTomq7gdmq%0AejVwM/D7426oJGl0o9y5nwPMV9VDVfUkcCOwebhCVd1eVd/tDu8A1o23mZKkPkYJ97XAw0PH+7qy%0A53I58A+HO5Fka5K5JHMLCwujt1KS1MtYJ1ST/DIwC2w73Pmqur6qZqtqdmZmZpxvLUkasnqEOvuB%0A9UPH67qyZ0hyPvAB4Ger6nvjaZ4krTxNTKgCdwEbk5yZ5CTgUmDHcIUkZwMfBy6pqkfH30xJUh9L%0AhntVHQSuAHYBDwI3VdUDSa5NcklXbRvwIuAzSe5JsuM5Xk6Spt40fEJ1lGEZqmonsHNR2TVDz88f%0Ac7skSc+Dn1CVpAYZ7pLUUysTqpKkKWO4S1JPU3DjbrhLUosMd0lqkOEuST3VFMyoGu6S1CDDXZJ6%0AWvn37Ya7JDXJcJekBhnuktTTFMynGu6S1CLDXZJ6W/m37oa7JDXIcJekBhnuktSTE6qSpIkw3CWp%0AQYa7JPU0BaMyhrsktchwl6SenFCVJE2E4S5JDTLcJaknd2KSJE2E4S5JPa38+3bDXZKaZLhLUoNW%0AT7oBkrRSbd+9n2279vLI4wc449RTni6fgvlUw12SDmf77v1cfct9HHjq+wDsf/zAhFvUz0jDMkku%0ATLI3yXySqw5z/uQkn+7O35lkw7gbKknH0rZde58O9sXefcPn2b57/zFuUT9Zar1mklXAF4E3AfuA%0Au4AtVbVnqM6vA6+uql9Lcinwc1X1i0d63dnZ2Zqbm+vV2O2793PlZ+7hqR/0+jJJWnFOXn0CH3rb%0Aq3nr2Wt7fV2Su6tqdql6o9y5nwPMV9VDVfUkcCOweVGdzcBfds9vBt6YJH0avJTtu/fz3k8b7JLa%0A8L2DP+B9N92zbL8BjBLua4GHh473dWWHrVNVB4EngJeNo4GHbNu1d5wvJ0kT94Navmw7pkshk2xN%0AMpdkbmFhodfXPjJlkxmSNIrlyrZRwn0/sH7oeF1Xdtg6SVYDLwG+vfiFqur6qpqtqtmZmZleDR1e%0AhiRJrViubBsl3O8CNiY5M8lJwKXAjkV1dgCXdc9/Hvhsjfkv61x5wVnjfDlJmrgTsnzZtuQ696o6%0AmOQKYBewCvhEVT2Q5Fpgrqp2AH8OfCrJPPAdBj8AxurQjLKrZSS14GhXy4xqyaWQy+VolkJK0vFu%0AnEshJUlTxnCXpAYZ7pLUIMNdkhpkuEtSgya2WibJAvC1o/zyNcC3xticaeA1Hx+85uPD87nmH6uq%0AJT8FOrFwfz6SzI2yFKglXvPxwWs+PhyLa3ZYRpIaZLhLUoOmNdyvn3QDJsBrPj54zceHZb/mqRxz%0AlyQd2bTeuUuSjmDqwn2pzbqnVZL1SW5PsifJA0ne05WfluSfk3yp+/elXXmS/FH3fbg3yWsmewVH%0AJ8mqJLuT3Nodn9ltsj7fbbp+UlfexCbsSU5NcnOSLyR5MMnrjoM+/s3u//T9SW5I8oIW+znJJ5I8%0AmuT+obLefZvksq7+l5Jcdrj3GsVUhXu3Wfd1wEXAJmBLkk2TbdXYHATeX1WbgHOBd3XXdhVwW1Vt%0ABG7rjmHwPdjYPbYCHzv2TR6L9wAPDh1/CPhIVf048BhweVd+OfBYV/6Rrt40+ijwj1X1E8BPMbj2%0AZvs4yVrgN4DZqnoVgz8bfilt9vMngQsXlfXq2ySnAR8EXstg/+oPHvqB0FtVTc0DeB2wa+j4auDq%0ASbdrma7174A3AXuB07uy04G93fOPA1uG6j9db1oeDHb1ug14A3ArEAYf7Fi9uL8Z7Cfwuu756q5e%0AJn0NPa/3JcBXFre78T4+tL/yaV2/3Qpc0Go/AxuA+4+2b4EtwMeHyp9Rr89jqu7cGW2z7qnX/Sp6%0ANnAn8PKq+kZ36pvAy7vnLXwv/hD4LeDQ9isvAx6vwSbr8MxrWvZN2I+BM4EF4C+6oag/S/JCGu7j%0AqtoP/AHwdeAbDPrtbtru52F9+3ZsfT5t4d68JC8C/hZ4b1X99/C5Gvwob2J5U5K3AI9W1d2Tbssx%0AtBp4DfCxqjob+B/+/9d0oK0+BuiGFDYz+MF2BvBCnj10cVw41n07beE+ymbdUyvJiQyC/a+r6pau%0A+L+SnN6dPx14tCuf9u/F64FLknwVuJHB0MxHgVO7Tdbhmdc00ibsK9w+YF9V3dkd38wg7FvtY4Dz%0Aga9U1UJVPQXcwqDvW+7nYX37dmx9Pm3hPspm3VMpSRjsRftgVX146NTw5uOXMRiLP1T+K92s+7nA%0AE0O//q14VXV1Va2rqg0M+vGzVfVLwO0MNlmHZ1/vsm7Cvtyq6pvAw0kO7Yj8RmAPjfZx5+vAuUl+%0AqPs/fuiam+3nRfr27S7gzUle2v3W8+aurL9JT0AcxYTFxcAXgS8DH5h0e8Z4XT/D4Fe2e4F7usfF%0ADMYbbwO+BPwLcFpXPwxWDn0ZuI/BaoSJX8dRXvt5wK3d81cA/wHMA58BTu7KX9Adz3fnXzHpdh/l%0Atf40MNf183bgpa33MfC7wBeA+4FPASe32M/ADQzmFZ5i8Fva5UfTt8Cvdtc/D7zjaNvjJ1QlqUHT%0ANiwjSRqB4S5JDTLcJalBhrskNchwl6QGGe6S1CDDXZIaZLhLUoP+D/Ww2REjQ9voAAAAAElFTkSu%0AQmCC)

```python
import json

with open('./imagenet_class_index.json','rb') as f:
    data = json.load(f)

print('predicted class, ',data["945"][1])
"""
predicted class,  bell_pepper
"""
```

### 对抗网络

![](images/对抗.JPG)

```python
import json
from PIL import Image
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as trans

# 定义归一化接口
class Normalize(nn.Module):
    def __init__(self,device=torch.device('cpu')):
        super(Normalize,self).__init__()
        self.mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    def forward(self,x):
        if x.ndimension() ==3:
            x = x.unsqueeze(0) # 3x224x224 ==> 1x3x224x224
        o = (x-self.mean)/self.std
        return o
class Index2Class:
    def __init__(self,filename='imagenet_class_index.json'):
        with open(filename,'rb') as f:
            self.data = json.load(f)
    def __getitem__(self,idx):
        idx = str(idx)
        return self.data[idx][1]
```

加载模型和测试图片

- 模型：vgg16
- 图片：辣椒

```python
model = nn.Sequential(Normalize(),
                     models.vgg16(pretrained=True))
img = Image.open('./figs/green-peper.jpg').resize((224,224))
x = trans.ToTensor()(img)
Y_TRUE = 945
y = torch.LongTensor([Y_TRUE])

plt.imshow(x.permute([1,2,0]).numpy());
```

![](images/辣椒.JPG)

FGSM 攻击

FGSM 攻击法由 GoodFellow 在 2016 年提出[文章链接](https://arxiv.org/abs/1412.6572) ，是最简单有效的一种攻击手段。假设 $J$ 是损失函数，那么所需的对抗性样本由下面公式给出：
$$
x_{adv}=x+\varepsilon \mathbb{sign}(\triangledown J(f(x),y_{true}))~,
$$
这里 $J$ 的选法有很多，常见的如交叉熵等等。

下面，构造一个对抗性“辣椒🌶”来欺骗神经网络。

- 计算梯度

```python
model.eval()
x.requires_grad = True
logit = model(x)
loss = F.cross_entropy(logit,y)
loss.backward()
print(x.grad.shape)
torch.Size([3, 224, 224])
```

- 沿着梯度符号方向攻击，生成对抗性样本

```python
epsilon = 5/255.0
x_adv = x.data + epsilon * x.grad.data.sign()
x_adv = torch.clamp(x_adv,0,1)
```

- 查看对抗性样本的分类结果

  方尖碑被网络识别成灯塔，但是人眼几乎感觉不到差别。

```python
logit = model(x_adv)
prob_adv = F.softmax(logit,dim=1).squeeze().data.numpy()

logit = model(x)
prob = F.softmax(logit,dim=1).squeeze().data.numpy()

plt.plot(prob,'-o',label='clean image')
plt.plot(prob_adv,'-*',label='adv image')

plt.legend()
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAD8CAYAAACMwORRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAHblJREFUeJzt3Xt0VPW99/H3lwQIKnKNFgmW2If7zUhAKLaC1YLWA6gn%0ABU5Pq7WPPK3ah2JXu7xVqfq01baeoz1ayipK7bKoaKXUcg49iHKqFiWIogSQaCmEeokIKMglge/z%0Ax56JQ5gkM8kkM3vn81prVmZfZu/vnp35zJ7fvpm7IyIi0dIh2wWIiEjmKdxFRCJI4S4iEkEKdxGR%0ACFK4i4hEkMJdRCSCFO4iIhGkcBcRiSCFu4hIBOVna8a9e/f2/v37Z2v2IiKhtG7duvfdvbCp8bIW%0A7v3796e8vDxbsxcRCSUz+3sq46lZRkQkghTuIiIRpHAXEYmgrLW5J1NTU0NVVRUHDx7MdintUkFB%0AAUVFRXTs2DHbpYhICzUZ7mb2AHAx8J67D08y3IB7gIuAj4Er3P3l5hRTVVVF165d6d+/P8Fkpa24%0AO7t27aKqqori4uJslyMiLZRKs8wiYEojwy8EBsQes4FfNreYgwcP0qtXLwV7FpgZvXr10q8mkQYs%0AXb+TCT9ZRfH1f2LCT1axdP3ObJfUqCbD3d3/B/igkVGmAQ95YA3Q3cz6NLcgBXv26L0XSW7p+p3c%0A8PvX2LnnAA7s3HOAG37/Wk4HfCZ2qPYFdiR0V8X6iYhEwk9XbOFAzZFj+h2oOcJPV2zJUkVNa9Oj%0AZcxstpmVm1l5dXV1W866RebNm8fPfvazVp3HRRddxJ49e1p1HiLSPP/YcyCt/rkgE0fL7AT6JXQX%0Axfodx90XAAsASktLW3xn7qXrd/LTFVv4x54DnNa9C9+bPIjpJeH80bB8+fJslyAiDTitexd2Jgny%0A07p3yUI1qcnElvsy4GsWGAfsdfe3MzDdRrVWG9hDDz3EyJEjGTVqFF/96lePG/7mm28yZcoURo8e%0Azec+9zk2b94MwB//+EfOPvtsSkpKOP/883n33XeBYKv/yiuvZOLEiZxxxhnce++9Sefbv39/3n//%0AfbZt28bgwYO54oorGDhwIF/5yldYuXIlEyZMYMCAAbz00ksAvPTSS4wfP56SkhI++9nPsmVL8PPw%0A448/5stf/jJDhw7lkksu4eyzz667zMOf//xnxo8fz1lnnUVZWRn79u1r0Xsl0l58b/IgunTMO6Zf%0Al455fG/yoCxV1LRUDoVcDEwEeptZFXAr0BHA3ecDywkOg6wkOBTy65ko7Id/3EjFPz5scPj67Xs4%0AfOToMf0O1Bzh+49vYPFL25O+ZuhpJ3PrPw1rcJobN27kjjvu4IUXXqB379588MHx+5Fnz57N/Pnz%0AGTBgAC+++CJXX301q1at4pxzzmHNmjWYGb/+9a+56667+PnPfw7A5s2beeaZZ/joo48YNGgQ3/rW%0Atxo9lryyspIlS5bwwAMPMGbMGH73u9/x3HPPsWzZMn70ox+xdOlSBg8ezF/+8hfy8/NZuXIlN954%0AI0888QT3338/PXr0oKKigtdff50zzzwTgPfff5877riDlStXcuKJJ3LnnXdy9913c8sttzRYh4gE%0A4i0C33n0FQD6hqCloMlwd/dZTQx34JqMVZSi+sHeVP9UrFq1irKyMnr37g1Az549jxm+b98+Xnjh%0ABcrKyur6HTp0CAiO0Z8xYwZvv/02hw8fPuZY8S996Ut07tyZzp07c8opp/Duu+9SVFTUYB3FxcWM%0AGDECgGHDhvGFL3wBM2PEiBFs27YNgL1793L55ZezdetWzIyamhoAnnvuOebMmQPA8OHDGTlyJABr%0A1qyhoqKCCRMmBO/T4cOMHz++2e+VSHszvaRvXbg/f/15Wa6maTl1hmqixrawASb8ZFXSNrC+3bvw%0A6P9pndA6evQo3bt355VXXjlu2Le//W2uu+46pk6dyrPPPsu8efPqhnXu3LnueV5eHrW1tY3OJ3H8%0ADh061HV36NCh7rU/+MEPmDRpEk8++STbtm1j4sSJjU7T3bngggtYvHhxU4spIhEQ2mvLtEYb2Hnn%0AnceSJUvYtWsXwHHNMieffDLFxcUsWbIECALz1VdfBYIt6b59g59ov/nNb5pdQ6oS57do0aK6/hMm%0ATOCxxx4DoKKigtdeew2AcePG8fzzz1NZWQnA/v37eeONN1q9ThHJjtCG+/SSvvz40hH07d4FI9hi%0A//GlI1rUBjZs2DBuuukmzj33XEaNGsV111133DgPP/wwCxcuZNSoUQwbNow//OEPQLDjtKysjNGj%0AR9c167Sm73//+9xwww2UlJQc80vg6quvprq6mqFDh3LzzTczbNgwunXrRmFhIYsWLWLWrFmMHDmS%0A8ePH1+0MFpHosaDJvO2VlpZ6/Zt1bNq0iSFDhmSlnqg4cuQINTU1FBQU8Oabb3L++eezZcsWOnXq%0A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%0ASUVORK5CYII=)

```python
idx2class = Index2Class()
idx = np.argmax(prob_adv)
print('adversarial example is classified as %s'%(idx2class[idx]))
adversarial example is classified as bath_towel
```

- 可视化对抗样本

```python
x_np = x_adv.numpy().transpose([1,2,0])
plt.imshow(x_np);
plt.axis('off');
```

![](images/辣椒2.JPG)

有目标的攻击

前面的攻击中，只需要识别错误即可。如果希望网络将 $x$ 识别成一个指定的 $y_{wrong}$ 该怎么做，一个办法就是去极小化输出的类别和目标类别的误差，即：
$$
\min_{{\|x'-x\|}_{\infty}\leqslant\varepsilon}J(f(x'),y_{wrong})
$$
和前面类似，可以使用下面的一步攻击法：
$$
x'=x-\alpha \mathbb{sign}(\triangledown J(x,y_{wrong}))
$$

```python
# 指定错误类别
Y_WRONG = 120
y = torch.LongTensor([Y_WRONG])

# 计算导数
model.eval()
x.grad.zero_()
logit = model(x)
loss = F.cross_entropy(logit,y)
loss.backward()

# attack
epsilon = 20/255.0
x_adv = x.data - epsilon * x.grad.data.sign()
x_adv = torch.clamp(x_adv,0,1)

logit = model(x_adv)
prob_adv = F.softmax(logit,dim=1).squeeze().data.numpy()

logit = model(x)
prob = F.softmax(logit,dim=1).squeeze().data.numpy()

plt.plot(prob,'-o',label='clean image')
plt.plot(prob_adv,'-*',label='adv image')
plt.legend()
```

![](data:image/png;base64,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)

```python
idx_adv = np.argmax(prob_adv)
print('The predicted label of adversarial example is %d, which is %s'%(idx_adv,idx2class[idx_adv]))
"""
The predicted label of adversarial example is 55, which is green_snake
"""
```

通过测试不同扰动大小 $\varepsilon$，可以看到，并没有成功的使得所生成对抗性样本被判别为我们所指定的类别。

- 这可能是单步方法并没有将优化问题解到最优

多步攻击

前面的单步有目标攻击并没有成功，原因是单步无法使得损失函数 $J$ 下降得非常小。为了克服这个困难， 我们可以将前面的迭代多进行几步，即
$$
x_{t+1} = clip_{\varepsilon}(x_t-\alpha~ sign(\triangledown J (x_t,y_{wrong})))
$$
其中 $t=1,2,\dots,T$ ，$clip_{\varepsilon}$ 是一个投影操作，保证整体 $\|x_{t+1}-x \|_{\infty} \leqslant \varepsilon$ 。

```python
def multistep_targeted_attack(net,x,y_target,alpha=2/255.0,epsilon=10/255.0,nsteps=5):
    net.eval()
    y_target = torch.LongTensor([y_target])
    J = nn.CrossEntropyLoss()
    x_adv = x.data.clone()
    x_adv.requires_grad = True
    for i in range(nsteps):
        # 计算导数
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        logit = net(x_adv)
        loss = J(logit,y_target)
        loss.backward()
        grad_sign = x_adv.grad.data.sign()

        # 一步攻击
        x_adv.data = torch.clamp(x_adv.data - alpha * grad_sign,0,1)
        dx = torch.clamp(x_adv.data - x.data,-epsilon,epsilon)
        x_adv.data = x.data + dx

        _,pred_label = torch.max(logit.data.view(-1),dim=0)
        print('%d-th, loss: %.2e, pred_label: %d'%(i+1,loss.item(),pred_label))
    return x_adv.data
x_adv = multistep_targeted_attack(model,x,Y_WRONG,
                                 alpha=1/255.0,
                                 epsilon=3/255.0,
                                 nsteps=10)
"""
1-th, loss: 2.17e+01, pred_label: 945
2-th, loss: 1.48e+01, pred_label: 936
3-th, loss: 1.01e+01, pred_label: 968
4-th, loss: 6.69e+00, pred_label: 929
5-th, loss: 4.42e+00, pred_label: 945
6-th, loss: 3.28e+00, pred_label: 317
7-th, loss: 4.00e+00, pred_label: 945
8-th, loss: 1.03e+00, pred_label: 120
9-th, loss: 5.15e-01, pred_label: 120
10-th, loss: 2.01e-01, pred_label: 120
"""
```

```python
logit = model(x_adv)
prob_adv = F.softmax(logit,dim=1).data.squeeze().numpy()
idx_adv = np.argmax(prob_adv)

label_pred = idx2class[idx_adv]
label_spec = idx2class[Y_WRONG]


plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(x_adv.numpy().transpose([1,2,0]));
plt.axis('off')
plt.title('predicted as  '+label_pred+', the specificed label is '+ label_spec);

plt.subplot(122)
plt.plot(prob_adv,'-o')
```

![](images/对抗2.JPG)

可视化对抗扰动

```python
dx = x_adv.data - x.data
dx = (dx-dx.min())/(dx.max()-dx.min())
dx = dx.numpy().transpose([1,2,0])
plt.imshow(dx)
```

![](images/对抗3.JPG)

### 迁移学习

在很多实际应用场景中，从头重新训练一个网络是没有必要的。譬如我要学一个分类狗的不同品种的分类器，如果从头训练的话代价太大。而一个有效的办法就是，使用ImageNet上预训练好的模型做初始化，在上面做微调（finetuning）。这样做不但会大大加快收敛，更重要的是往往能够有更高的测试准确率。这是因为，我们所要分类的目标和ImageNet不一样，但是都是自然图片，往往共享很多类似的特征。因此在很多，数据标记数据很难获得的情形下，使用预训练模型是一个非常重要的方法。

上面的应用方式，我们有两个domain，一个是ImageNet我们一般称为source domain；一个是我们实际要解决的问题，称为target domain。这两个domain的数据相似而不同，如何利用 source domain所学到的知识去帮助target domain的学习，被统称为迁移学习（transfer learning）。这里我们采用的是一种最简单的迁移学习方法：微调（finetuning）。更多详细的资料可参考下面链接：

- https://jindongwang.github.io/transferlearning/

### 训练一个识花器

这里，我们利用预训练的模型，训练一个识花器。数据集使用的是Oxford102, Oxford102是一个包含了102种花的数据集（[链接](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)）。数据集的信息总结如下

| 类别数 | 训练集大小 | 验证集大小 | 测试集大小 |
| :----: | :--------: | :--------: | :--------: |
|  102   |    1020    |    1020    |    6149    |

一个102类，训练集每类有10张图片，而测试集中每个类别的样本数目不固定。

```python
import time
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
import torchvision.transforms as trans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

用提供好的 python 脚本下载数据集

训练集，验证集和测试集分别存储在：

- ./data/oxford102/train.txt
- ./data/oxford102/valid.txt
- ./data/oxford102/test.txt

每个文件的每行有两个元素，分别是图片的绝对路径和图片对应的类别：

```
path_to_image		image_label
```

![](images/todo0.JPG)

![](images/todo1.jpg)

![](images/todo2.jpg)

![](images/todo3.jpg)

![](images/todo4.jpg)

![](images/todo5.jpg)

![](images/todo6.jpg)

![](images/todo7.jpg)

![](images/todo8.jpg)

## 文字生成

使用RNN对文字生成过程进行建模。我们这里面实践的是character-level的生成模型，更好的做法是在word-level进行建模。

![](images/文字生成.jpg)

```python
# 加载相关包
import time
import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.optim import lr_scheduler 

# 加载数据
text = unidecode.unidecode(open('shakespeare.txt').read())
text_len = len(text)
print(text[0:100])
"""
First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou
"""
# 总文本的字符数
print(text_len)
"""1115394"""
# 构造字典
all_characters = string.printable
n_characters = len(all_characters)
print(all_characters)
"""
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
"""

# 随机生成一段
def random_chunk(chunk_len=200):
    idx_start = random.randint(0,text_len-chunk_len)
    idx_end = idx_start + chunk_len
    return text[idx_start:idx_end]

random_chunk()
# 结果
# "y master?\n\nFirst Servingman:\nNay, it's no matter for that.\n\nSecond Servingman:\nWorth six on him.\n\nFirst Servingman:\nNay, not so neither: but I take him to be the\ngreater soldier.\n\nSecond Servingman:\nF"
```

```python
# 把字符串转化成张量
# 需要是LongTensor张量，因为nn.Embedding只接受LongTensor作为输入
def char2tensor(string):
    n = len(string)
    res = torch.LongTensor(n)
    for i in range(n):
        res[i] = all_characters.index(string[i])
    return res
char2tensor(random_chunk())
"""
tensor([18, 21, 94, 10, 23, 13, 94, 24, 23, 94, 22, 18, 23, 14, 73, 96, 54, 17,
        14, 94, 28, 17, 24, 30, 21, 13, 94, 23, 24, 29, 94, 31, 18, 28, 18, 29,
        94, 34, 24, 30, 75, 96, 96, 47, 40, 50, 49, 55, 40, 54, 77, 96, 58, 17,
        10, 29, 73, 94, 12, 10, 23, 28, 29, 94, 23, 24, 29, 94, 27, 30, 21, 14,
        94, 17, 14, 27, 82, 96, 96, 51, 36, 56, 47, 44, 49, 36, 77, 96, 41, 27,
        24, 22, 94, 10, 21, 21, 94, 13, 18, 28, 17, 24, 23, 14, 28, 29, 34, 94,
        17, 14, 94, 12, 10, 23, 77, 94, 18, 23, 94, 29, 17, 18, 28, 73, 96, 56,
        23, 21, 14, 28, 28, 94, 17, 14, 94, 29, 10, 20, 14, 94, 29, 17, 14, 94,
        12, 24, 30, 27, 28, 14, 94, 29, 17, 10, 29, 94, 34, 24, 30, 94, 17, 10,
        31, 14, 94, 13, 24, 23, 14, 73, 96, 38, 24, 22, 22, 18, 29, 94, 22, 14,
        94, 15, 24, 27, 94, 12, 24, 22, 22, 18, 29, 29, 18, 23, 16, 94, 17, 24,
        23, 24])
"""
```

### 定义模型

我们这里面使用了

- `nn.Embedding` 把one-hot编码转化成实值向量
- `nn.LSTM` 用来编码序列
- `nn.Linear` 做预测

$$
\begin{aligned}
& i_t = \sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi})\\
& f_t = \sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})\\
& g_t = \tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})\\
& o_t = \sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})\\
& c_t = f_tc_{t-1}+i_tg_t\\
& h_t = o_t\tanh(c_t)
\end{aligned}
$$

```python
class CharRNN(nn.Module):
    def __init__(self,dict_size,embed_size,hidden_size,num_layers=1):
        super(CharRNN,self).__init__()
        self.embed = nn.Embedding(dict_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.fc = nn.Linear(hidden_size,dict_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self,x,h0,c0):
        # seq_len x batch_size 
        o = self.embed(x) 
        # seq_len x batch_size x dict_size
        o,(h,c) = self.lstm(o,(h0,c0))
        # seq_len x batch_size x hidden_size
        o = o.view(-1,self.hidden_size)
        o = self.fc(o)
        # seq_len*batch_size x dict_size
        return o,h,c

    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))
```

![](images/todo11.JPG)

![](images/todo12.JPG)

![](images/todo13.JPG)

![](images/todo14.JPG)

![](images/todo15.JPG)

![](images/todo16.JPG)

![](images/todo17.JPG)

![](images/todo18.JPG)

