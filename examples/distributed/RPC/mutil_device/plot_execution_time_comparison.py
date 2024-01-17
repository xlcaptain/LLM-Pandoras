import matplotlib.pyplot as plt
import random

# 初始化列表来存储数据
num_splits = [1, 2, 4, 8, 16, 32]
execution_times1 = [3.811338186264038, 0.35001182556152344, 0.32590293884277344, 0.3609011173248291, 0.3617372512817383, 0.3465299606323242]
execution_times2 = [47.41341423988342, 40.9799530506134, 35.238571643829346, 27.33722448348999, 23.572229385375977, 27.643784284591675]
execution_times3 = [56.9645037651062, 48.93175721168518, 49.85330581665039, 55.699986934661865, 56.100653409957886, 56.08349418640137]

# 绘制三条线
plt.plot(num_splits, execution_times1, marker='o', label='single process')
plt.plot(num_splits, execution_times2, marker='*', label='multi-process')
plt.plot(num_splits, execution_times3, marker='^', label='multi-device')

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.xlabel('Number of splits')
plt.ylabel('Execution time (s)')
plt.title('Execution time vs Number of splits')
plt.grid(True)

# 保存图形
plt.savefig('execution_time_vs_num_splits.png')

# 显示图形
plt.show()
