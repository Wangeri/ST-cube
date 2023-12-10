# coding=utf-8
# Filename  :空间合并
# Time      :2022/11/2

import copy
import math
import os
import time

import jenkspy

from osgeo import gdal
import numpy as np
from Function_Set import read_tif, write_tif
import gc


class SpaceCube:
    def __init__(self, data, sh, id=-1):
        # 假设需要读图
        self.im = data
        # 分类合并矩阵
        self.clas = np.zeros(self.im.shape, dtype=np.int32) - 1  # 节点编号数据（未编号为-1）
        self.id = id  # 赋值节点编号-1
        self.idmin = id + 1
        self.arr = []
        self.sh = sh

    # 用于判断特征是否能被合并成一个立方体
    def judge_similar(self, x, y, zc, bc, add, bc1):
        temp = [x] + y
        std = np.std(np.array(temp), ddof=0)
        # 计算周长与最小边长的关系
        h_color = std * 1 * len(temp)
        h_cpt = math.sqrt(len(temp)) * 1 * zc - math.sqrt(len(temp) - 1) * (zc - add) - 4
        h_smooth = len(temp) * zc / bc - 1 - (len(temp) - 1) * (zc - add) / bc1
        if x > 900 and y[0] > 900:
            if len(y) > 60000:
                return False
            else:
                return True
        if h_color * 0.9 + (h_cpt * 0.5 + h_smooth * 0.5) * 0.1 < self.sh:
            return True
        else:
            self.arr.append([x, y[0], zc, h_color * 0.9, h_cpt * 0.05, h_smooth * 0.05])
            return False

    # 计算新增一个像元需要加多少周长
    def clzc(self, i, j):
        xb = [[], []]
        # 上面一行有元素
        if i - 1 >= 0:
            xb[0].append(i - 1)
            xb[1].append(j)
        # 下面一行有元素
        if i + 1 <= self.clas.shape[0] - 1:
            xb[0].append(i + 1)
            xb[1].append(j)
        if j - 1 >= 0:
            xb[0].append(i)
            xb[1].append(j - 1)
        if j + 1 <= self.clas.shape[1] - 1:
            xb[0].append(i)
            xb[1].append(j + 1)
        xb = (tuple(xb[0]), tuple(xb[1]))
        # print(xb)
        judge = self.clas[xb] == self.clas[i, j]
        # print(judge)
        return 4 - 2 * np.sum(judge == 1)

    # 种子填充算法
    def seed_fill(self, i, j):
        self.clas[i, j] = self.id
        # 光谱值
        typ = [self.im[i, j]]
        # 标号像素待编号list
        plist = [[i, j]]
        # index的最大和最小值, 用来算可能周长
        max_i = i
        min_i = i
        min_j = j
        max_j = j
        zc = 4
        # 循环至list为空
        while plist:
            # 判断是否在边界, 上侧
            if plist[0][0] == 0:
                # 周边像素
                x = plist[0][0] + 1
                y = plist[0][1]
                # 判断是否未添加到list且需要编号
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    max_i += 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # 判断过没编号的像素点标-2（防止重复添加）
                        # self.clas[x, y] = -2
                        # 添加到待编号list
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        max_i -= 1
            # 下同 下侧
            elif plist[0][0] == self.im.shape[0] - 1:
                x = plist[0][0] - 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    min_i -= 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        min_i += 1
            else:
                x = plist[0][0] - 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    min_i -= 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        min_i += 1
                x = plist[0][0] + 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    max_i += 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        max_i -= 1
            # 左侧
            if plist[0][1] == 0:
                x = plist[0][0]
                y = plist[0][1] + 1
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    max_j += 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        max_j -= 1
            # 右侧
            elif plist[0][1] == self.im.shape[1] - 1:
                x = plist[0][0]
                y = plist[0][1] - 1
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    min_j -= 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        min_j += 1
            else:
                x = plist[0][0]
                y = plist[0][1] - 1
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    min_j -= 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        min_j += 1
                x = plist[0][0]
                y = plist[0][1] + 1
                if self.clas[x, y] == -1:
                    self.clas[x, y] = self.clas[i, j]
                    add = self.clzc(x, y)
                    zc += add
                    bc1 = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    max_j += 1
                    bc = (max_j - min_j + max_i - min_i + 1 + 1) * 2
                    if self.judge_similar(self.im[x, y], typ, zc, bc, add, bc1):
                        # self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                    else:
                        self.clas[x, y] = -1
                        zc -= add
                        max_j -= 1
            # 标上编号
            self.clas[plist[0][0], plist[0][1]] = self.id
            # 移除已判断像素点
            plist.pop(0)
            print(f'\r{i}/{self.im.shape[0]},{j}/{self.im.shape[1]} {len(plist)}, {len(typ)},{x, y}', end='')

    def classify(self):
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                if self.clas[i, j] == -1:
                    self.id += 1
                    self.seed_fill(i, j)
                    # self.fe.append([self.id, self.im[i, j]])
        return self.clas

    def save_clas(self, path):
        np.save(path, self.clas)


# 把时间序列的数据转换为时间的矩阵
def change_data():
    index = np.load('./data/riceindex.npy')
    cancha = np.load('./data/riceinerlist.npy')[:, 2:]
    for j in range(cancha.shape[1]):
        reduction = np.full((np.max(index[:, 0]) + 1, np.max(index[:, 1]) + 1), np.nan)
        for i in range(index.shape[0]):
            value = cancha[i][j]
            x = index[i][0]
            y = index[i][1]
            reduction[x][y] = value
        np.save(f'./data/original/{j}.npy', reduction)


# def get_value():
#     ttt = np.zeros(img.clas.shape)
#     for i in range(np.max(classify)):
#         x, y = np.where(img.clas == i)
#         temp = []
#         for j in range(len(x)):
#             temp.append(img.im[x[j], y[j]])
#         value = np.mean(np.array(temp))
#         for k in range(len(x)):
#             ttt[x[k], y[k]] = value
#         ttt[ttt < 0] = np.nan
#         ttt[ttt > 1] = np.nan

def clas_stgs(clas, i):
    # 计算stgs，首先将classify处理一下，得到对应的list，合起来计算list包含三个东西分别是[std, mean_value, count]
    obj = {}
    classify = clas
    value = np.load(f'./data/{i}.npy')
    for i in range(classify.shape[0]):
        for j in range(classify.shape[1]):
            if not np.isnan(value[i, j]):
                if obj.get(classify[i][j]):
                    obj[classify[i, j]]['coordinate'].append([i, j])
                    obj[classify[i, j]]['value'].append(value[i, j])
                else:
                    obj[classify[i, j]] = {'id': classify[i, j], 'coordinate': [[i, j]], 'value': [value[i, j]]}
    arr = []
    for item in obj:
        value = obj[item]['value']
        if len(value) > 1:
            arr.append([np.std(value), np.mean(value), len(value)])
    astd = np.std(np.array(arr).transpose()[1])
    count = np.sum(np.array(arr).transpose()[2])
    mc = np.mean(np.array(arr).transpose()[2])
    stgs = 0
    # np.save('./save_npy/arr.npy', arr)
    for i in arr:
        stgs += (1 - i[0] / astd) * i[2]
    return stgs / count, mc


def filter_npy(path):
    rice = np.load(path)
    rice[np.isnan(rice)] = 999
    rice[rice > 0.7] = 999
    rice[rice < 0.1] = 999
    img = SpaceCube(rice, sh)
    img.classify()
    classify = img.clas
    return classify

