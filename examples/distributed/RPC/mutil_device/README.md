# 分布式训练与单卡训练对比实验

本项目展示了如何使用PyTorch的RPC框架实现ResNet50模型的分布式训练，并提供了将分布式训练改造为单卡训练的方法，以便进行性能对比。

## 环境要求

- Python 3.9
- PyTorch 2.1
- torchvision
- matplotlib

确保你的环境中安装了以上依赖。

## 文件结构

- `complete_resnet_demo.py`: 包含分布式训练的完整代码。
- `single_gpu_train.py`: 包含单卡训练的完整代码。
- `README.md`: 项目说明文件。

## 如何运行

首先，确保所有的依赖都已经安装。你可以使用以下命令安装所需的依赖：
```bash
pip install torch torchvision matplotlib
```


### 分布式训练
其中：
```angular2html
`--rank` 表示当前工作节点的编号，`--world_size` 表示总的工作节点数量，`--num_splits` 表示输入数据的分片数量。
```

运行分布式训练的代码，你需要在多个终端上启动不同的工作节点。例如，如果你有两个工作节点，你可以在两个终端上分别运行以下命令：
```
# 普通单卡训练
python single_gpu_resnet50.py

# 单节点多卡训练,具体相关参数请查看对应的py文件
python complete_resnet_demo.py
```

```angular2html
# 多节点多卡训练，首先需要修改主节点的进程数，我们只模拟两个节点。将nprocs修改为2，标识只需两个进程。其中rank会自动适配。之后运行该文件即可。

mp.spawn(run_worker, args=(world_size, num_split), nprocs=2, join=True)

# 由于 world_size=3,但是主节点只启动了两个进程，所以rpc.init_rpc将会阻塞，一直等待第三个进程的连接。即rank=2的连接。你需要在第二个节点进行init，并且放好对应的模型即可：

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.20.59'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp5s0'
    os.environ['TP_SOCKET_IFNAME'] = 'enp5s0'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    # block until all rpcs finish
    rpc.shutdown()

if __name__=="__main__":
    run_worker(2, 3)
```

## 性能对比

为了比较分布式训练和单卡训练的性能，你可以记录两种方式下的训练时间，并使用matplotlib绘制出来。代码中已经包含了绘图的部分，你可以直接查看或保存生成的图表。
<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/distributed/execution_time_vs_num_splits.png">

## 注意事项

- 确保在运行分布式训练之前，所有工作节点都能够相互通信。
- 分布式训练需要多个工作节点，而单卡训练只需要一个节点。
- 分布式训练的性能受到网络延迟、数据传输等多种因素的影响。

## 联系方式

如果你在使用过程中遇到任何问题，欢迎通过以下方式联系我们：

- 邮箱：[xulang@dnect.cn]
