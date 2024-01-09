# 分布式深度学习

这个代码示例展示了如何使用PyTorch进行分布数据并行训练。
我们将从头开始展示如何实现分布式数据并行，在这过程中，你可能需要拥有多块GPU来进行实验。如果没有GPU请特别注意模型训练部分可能需要改动。

## 导入所需的库

```angular2html
import os
import torch
from random import Random
from torchvision import datasets, transforms
import torch.distributed as dist
from math import ceil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
import torch.multiprocessing as mp
```


## 定义一个简单的CNN模型


```angular2html
python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).init()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 7 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 7 7)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```


## 数据切分到不同的GPU上

```angular2html
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
```

这段代码通过默认的size参数可以被用来将数据集分割为训练集、验证集和测试集的。也可以传入参数来进行数据集切分。这在机器学习中是常见的做法，用于评估模型的性能。

这里有两个类：Partition 和 DataPartitioner。

Partition 类用于创建数据的子集，它接收数据和索引，然后通过索引来获取数据。

DataPartitioner 类用于将数据分割为多个部分。它接收数据、分割比例和随机种子。然后，它将数据的索引打乱并根据给定的比例分割索引。最后，它使用这些索引来创建数据的子集。

下面是如何使用这段代码的一个例子：


```angular2html
from random import Random

# 假设我们有一些数据
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建一个数据分割器
partitioner = DataPartitioner(data)

# 获取训练集
train = partitioner.use(0)
print('Training set:', [train[i] for i in range(len(train))])

# 获取验证集
val = partitioner.use(1)
print('Validation set:', [val[i] for i in range(len(val))])

# 获取测试集
test = partitioner.use(2)
print('Test set:', [test[i] for i in range(len(test))])
```
输出结果为：
```angular2html
Training set: [3, 9, 4, 6, 7, 5, 10]
Validation set: [1, 2]
Test set: [8]
```
上面的数据切分是数据并行的核心之一。我们可能需要将整个数据集等份切分成多份分布到不同的GPU上面。