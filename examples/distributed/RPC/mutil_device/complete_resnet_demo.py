import os
import time

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from models import ResNetShard1, ResNetShard2

#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.
num_classes = 1000


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args=("cuda:1",) + args,
            kwargs=kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size):
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(split_size, ["worker1", "worker2"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = '192.168.20.59'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    os.environ['TP_SOCKET_IFNAME'] = 'eno1'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 3
    num_splits = [1, 2, 4, 8, 16, 32]
    execution_times = []
    for num_split in num_splits:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        execution_time = tok - tik
        execution_times.append(execution_time)
        print(f"number of splits = {num_split}, execution time = {execution_time}")

    # print(execution_times)

    # for num_split in [1, 2, 4, 8, 16, 32]:
    #     tik = time.time()
    #     mp.spawn(run_worker, args=(3, num_split), nprocs=2, join=True)
    #     tok = time.time()
    #     print(f"number of splits = {num_split}, execution time = {tok - tik}")
