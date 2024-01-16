# master 节点

import os
import torch
import torch.distributed.rpc as rpc

os.environ['MASTER_ADDR'] = '192.168.20.59'
os.environ['MASTER_PORT'] = '29500'
os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'  # 使用ifconfig，进行查看
os.environ['TP_SOCKET_IFNAME'] = 'eno1'

rpc.init_rpc(
    "master",
    rank=0,
    world_size=2,

)

fut1 = rpc.rpc_async("worker", torch.add, args=(torch.ones(2), 3))
fut2 = rpc.rpc_async("worker", min, args=(1, 2))
result = fut1.wait() + fut2.wait()

print("Result:", result)
rpc.shutdown()


# worker 节点
import os
from torch.distributed import rpc
os.environ['MASTER_ADDR'] = '192.168.20.59'
os.environ['MASTER_PORT'] = '29500'
os.environ['GLOO_SOCKET_IFNAME'] = 'eno1' # 当与master不同主机是，此处需要设置为对应主机的GLOO_SOCKET_IFNAME，同样ifconfig或者ip addr
os.environ['TP_SOCKET_IFNAME'] = 'eno1'
rpc.init_rpc(
    "worker",
    rank=1,
    world_size=2,
)

rpc.shutdown()
