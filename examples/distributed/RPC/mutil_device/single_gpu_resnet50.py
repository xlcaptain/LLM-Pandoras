import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, Bottleneck

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义完整的ResNet50模型
class MyResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(MyResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], *args, **kwargs)


def train(model, loss_fn, optimizer, num_batches, batch_size, image_w, image_h, num_classes):
    model.train()
    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size, 1).to(device)
    for i in range(num_batches):
        print(f"Processing batch {i}")
        # 生成随机输入和标签
        inputs = torch.randn(batch_size, 3, image_w, image_h).to(device)
        labels = torch.zeros(batch_size, num_classes).to(device).scatter_(1, one_hot_indices, 1)

        # 前向传播
        outputs = model(inputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


# 主函数
if __name__ == "__main__":
    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128
    num_classes = 1000
    num_splits = [1, 2, 4, 8, 16, 32]
    execution_times = []

    # 创建模型、损失函数和优化器
    model = MyResNet50(num_classes=num_classes).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # 训练模型并记录时间
    for num_split in num_splits:
        tik = time.time()
        train(model, loss_fn, optimizer, num_batches, batch_size, image_w, image_h, num_classes)
        tok = time.time()
        execution_time = tok - tik
        execution_times.append(execution_time)
        print(f"number of splits = {num_split}, execution time = {execution_time}")

    print(execution_times)
    # 绘制执行时间图
    plt.plot(num_splits, execution_times, marker='o', label='single GPU')
    plt.xlabel('Number of splits')
    plt.ylabel('Execution time (s)')
    plt.title('Execution time for different batch splits on a single GPU')
    plt.legend()
    plt.show()