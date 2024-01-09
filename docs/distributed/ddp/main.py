# -*-coding: utf-8 -*-

import os
from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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


# # 假设我们有一些数据
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# # 创建一个数据分割器
# partitioner = DataPartitioner(data)

# # 获取训练集
# train = partitioner.use(0)
# print('Training set:', [train[i] for i in range(len(train))])

# # 获取验证集
# val = partitioner.use(1)
# print('Validation set:', [val[i] for i in range(len(val))])

# # 获取测试集
# test = partitioner.use(2)
# print('Test set:', [test[i] for i in range(len(test))])


def partition_dataset():
    dataset = datasets.MNIST('/home/xulang/project/distributed/data', train=True, download=True,
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
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        # param.grad.data /= size

        # 创建一个接收缓冲区
        recv_buff = torch.zeros_like(param.grad.data)
        # 使用allreduce函数来进行梯度聚合
        allreduce(param.grad.data, recv_buff)
        # 使用接收缓冲区中的数据来更新梯度
        param.grad.data = recv_buff / size


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


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

    # 绘制不同GPU数量下的执行时间对比图
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times, marker='o')
    plt.title('Execution Time by Number of GPUs')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(sizes)
    plt.grid(True)
    plt.savefig('gpu_scaling2.png')
    plt.show()


