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

## 获取MNIST数据集
```angular2html
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz
```

## 分布式训练核心代码
``` diff
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()

    device = torch.device(f'cuda:{rank}')  # 指定每个进程的GPU
    torch.cuda.set_device(device)  # 设置当前进程的默认设备

    model = SimpleCNN().to(device) 
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = data.to(device), target.to(device)  # 将数据和目标移动到GPU
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
```

> [!NOTE]  
>  对于没有GPU资源的，我们需要修改代码。
> 
>```diff
> - device = torch.device(f'cuda:{rank}')  # 指定每个进程的GPU
> - torch.cuda.set_device(device)  # 设置当前进程的默认设备
> - model = SimpleCNN().to(device) 
> + model = SimpleCNN()
> - data, target = data.to(device), target.to(device)  # 将数据和目标移动到GPU
>```


## 初始化分布式环境
```angular2html
def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

```

## 训练
```angular2html
if __name__ == "__main__":
    from time import time
    import matplotlib.pyplot as plt
    sizes = [1, 2, 3, 4]  # 假设你想比较1, 2, 和 4个GPU的性能
    times = []
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    for size in sizes:
        start_time = time()
        processes = []
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        end_time = time()
        total_time = end_time - start_time
        times.append(total_time)
        print(f"Total execution time for {size} GPUs: {total_time} seconds")
```

## 结果显示
<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/distributed/gpu_scaling.png">
从图中可看出，随着GPU数量的增加，训练时间明显降低，但是当GPU越多时，似乎降低的时间开始衰减，这部分导致的原因可能之一是由于分布式归约所导致的，也就是梯度平均过程中的通信导致的。

## 扩展一：并行通信模式
<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/distributed/Collective_Communication.png">


## 扩展二：自定义环归约

```angular2html
""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]

```
这个函数实现了一个称为环归约（ring-Allreduce）的操作，这是一种在分布式计算中常用的并行通信模式。环归约是一种高效的方式来在所有进程间进行数据聚合，例如求和或者求平均。

在这个函数中，每个进程都有一个发送缓冲区（send_buff）和一个接收缓冲区（recv_buff）。每个进程将自己的数据发送给它的右邻居，并从它的左邻居接收数据。然后，每个进程将接收到的数据添加到一个累积缓冲区（accum）中。这个过程重复size - 1次，其中size是进程的总数。在所有的迭代完成后，accum中的数据就是所有进程的数据之和。

这个函数的主要用途是在分布式训练中进行梯度聚合。在数据并行的训练中，每个进程都会计算出模型参数的梯度。为了同步更新模型参数，所有的进程需要计算出梯度的全局平均值。这可以通过首先使用环归约来计算出梯度的全局和，然后再将这个和除以进程的数量来实现。

请注意，这个函数使用了非阻塞发送（dist.isend）和阻塞接收（dist.recv）。这意味着每个进程在发送数据后可以立即开始接收数据，而不需要等待发送完成。这可以提高通信的并行性和效率。

我们可以使用我们自定义的ring-Allreduce来替换掉之前我们使用默认的归约。
```diff
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
-       dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
-       param.grad.data /= size
    
+       # 创建一个接收缓冲区
+       recv_buff = torch.zeros_like(param.grad.data)
+       # 使用allreduce函数来进行梯度聚合
+       allreduce(param.grad.data, recv_buff)
+       # 使用接收缓冲区中的数据来更新梯度
+       param.grad.data = recv_buff / size
```

