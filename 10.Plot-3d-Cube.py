# _*_ coding: utf-8 _*_
import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Create axis
    axes = [21, 21, 21]
    # Create Data
    data = np.zeros(axes, dtype=np.bool_)
    # 从这里开始定义哪些地方填值
    # 首先来定义第一层的情况
    data[:, :, 0] = True
    size_list = [5, 4, 5, 5, 4, 4, 3, 5, 4, 10, 11, 9, 11, 12, 12, 11, 9, 7, 11, 5, 3]
    for i in range(21):
        # 限定x和y的范围
        rod = size_list[i]
        flag = 1
        for j in range(rod):
            count = math.ceil(j / 2)
            just = (rod - 1) / 2 if rod % 2 == 1 else rod / 2
            start = np.random.randint(10 - just, 10)
            end = np.random.randint(10, 10 + just + 1)
            print(i, rod, j, start, end, count, flag, 10 + count * flag)
            data[start:end, 10 + count * flag, i] = True
            flag = -flag
    sd_list = [[10, 11], [9, 11], [8, 12], [7, 17], [6, 18], [4, 18], [4, 17], [4, 18], [4, 19], [5, 17], [5, 17],
               [5, 16], [6, 15], [8, 13], [8, 12], [9, 12]]
    for i in range(len(sd_list)):
            for j in range(21):
                data[sd_list[i][0]:sd_list[i][1], i, j] = True
    #     data[sd_list[i][0]:sd_list[i][1], i, 20] = True
    standard = data[:, :, 0]
    for j in range(1, 20):
        for k in range(len(sd_list)):
            data[sd_list[k][0] - np.random.randint(1, 3):sd_list[k][1] + np.random.randint(-1, 2), k, j] = True

    data[9:12, 13, 0] = True

    # Control Transparency
    alpha = 1.0

    # Control colour
    # z小的时候，从中间开始向四周面积小，z到10的时候，面积最大，之后递减
    colors = np.empty(axes + [4], dtype=np.float32)
    list = ['#B0A3C2', '#ffffff']
    for i in range(len(list)):
        list[i] = [int(list[i][1:3], 16), int(list[i][3:5], 16), int(list[i][5, 7], 16), alpha]

    for i in range(21):
        colors[:, :, i] = list + [alpha]
        # colors[:, :, i] = list[i]

    # Plot figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.voxels(data, facecolors=colors, edgecolors=colors, linewidth=1)
    plt.axis('off')
    plt.show()
