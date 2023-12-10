# coding=utf-8
# Filename  :Judge-Sh
# Time      :2022/11/8


import copy
import math
import jenkspy

from osgeo import gdal
import numpy as np
from Function_Set import read_tif, write_tif


# 现在定义不同的sh，然后计算出0这个时刻sh = 10 15 20 30 40 50 60的情况
class SpaceCube:
    def __init__(self, data, sh, id=-1):
        # 假设需要读图
        self.im = data
        # 分类合并矩阵
        self.clas = np.zeros(self.im.shape, dtype=np.int32) - 1  # 节点编号数据（未编号为-1）
        self.id = id  # 赋值节点编号-1
        self.idmin = id + 1
        self.sh = sh

    # 用于判断特征是否能被合并成一个立方体
    def judge_similar(self, x, y):
        temp = [x]
        for i in y:
            temp.append(i)
        std = np.std(np.array(temp), ddof=0)
        if std * 1 * len(temp) < self.sh:
            return True
        else:
            return False

    # 种子填充算法
    def seed_fill(self, i, j):
        # 光谱值
        typ = [self.im[i, j]]
        # 标号像素待编号list
        plist = [[i, j]]
        # arr = []
        # 循环至list为空
        while plist:
            # 判断是否在边界
            if plist[0][0] == 0:
                # 周边像素
                x = plist[0][0] + 1
                y = plist[0][1]
                # 判断是否未添加到list且需要编号
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        # 判断过没编号的像素点标-2（防止重复添加）
                        self.clas[x, y] = -2
                        # 添加到待编号list
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            # 下同
            elif plist[0][0] == self.im.shape[0] - 1:
                x = plist[0][0] - 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            else:
                x = plist[0][0] - 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                x = plist[0][0] + 1
                y = plist[0][1]
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            if plist[0][1] == 0:
                x = plist[0][0]
                y = plist[0][1] + 1
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            elif plist[0][1] == self.im.shape[1] - 1:
                x = plist[0][0]
                y = plist[0][1] - 1
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            else:
                x = plist[0][0]
                y = plist[0][1] - 1
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
                x = plist[0][0]
                y = plist[0][1] + 1
                if self.clas[x, y] == -1:
                    if self.judge_similar(self.im[x, y], typ):
                        self.clas[x, y] = -2
                        typ.append(self.im[x, y])
                        plist.append([x, y])
            # 标上编号
            self.clas[plist[0][0], plist[0][1]] = self.id
            # 移除已判断像素点
            plist.pop(0)
            if len(plist) > 100:
                print(f'\r{len(plist)}, {len(typ)},{x, y}', end='')

    def classify(self):
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                print(f'\r{i}/{self.im.shape[0]},{j}/{self.im.shape[1]}', end='')
                if self.clas[i, j] == -1:
                    self.id += 1
                    self.seed_fill(i, j)
                    # self.fe.append([self.id, self.im[i, j]])
        return self.clas

    def save_clas(self, path):
        np.save(path, self.clas)


def test():
    for i in range(36, 43):
        print(i, 'start')
        rice = np.load(rf'./data/{i}.npy')
        rice[np.isnan(rice)] = 999
        img = SpaceCube(rice, sh=20)
        img.classify()
        classify = img.clas
        img.save_clas(f'./save_npy/classify{i}.npy')
        ttt = np.zeros(classify.shape)
        for m in range(np.max(classify)):
            x, y = np.where(classify == m)
            temp = []
            for j in range(len(x)):
                temp.append(img.im[x[j], y[j]])
            value = np.mean(np.array(temp))
            for k in range(len(x)):
                ttt[x[k], y[k]] = value
        ttt[ttt < 0] = np.nan
        ttt[ttt > 1] = np.nan
        np.save(f'./save_npy/mean_value{i}.npy', ttt)
        im_proj, im_Geotrans, im_data = read_tif('S2_20190408.tif')
        write_tif(f'./output_img/mean_value{i}.tif', ttt, im_Geotrans, im_proj, gdal.GDT_Float32)
        print(i, 'end')


def clas_stgs(clas):
    # 计算stgs，首先将classify处理一下，得到对应的list，合起来计算list包含三个东西分别是[std, mean_value, count]
    obj = {}
    classify = clas
    value = np.load(f'./data/0.npy')
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


if __name__ == "__main__":
    # s_cube = np.load('./time-cube/space-cube-dict0.npy', allow_pickle=True).item()
    classify = np.load('./save_npy/classify0.npy')
    clas_stgs(classify)
    # for data in [0, 21]:
    #     rice = np.load(f'./data/{data}.npy')
    #     rice[np.isnan(rice)] = 999
    #     result = [[5]]
    #     for i in range(1):
    #         sh = result[i][0]
    #         img = SpaceCube(rice, sh)
    #         classify = img.classify()
    #         stgs, mc = clas_stgs(classify)
    #         result[i].append(stgs)
    #         result[i].append(mc)
    #     print(data, result)

    # [0.0, 0.298, 0.364, 0.751]
    for sh in [5, 10, 20, 30, 40]:
        rice = np.load('./data/original/0.npy')
        img = SpaceCube(rice, sh)
        classify = img.classify()

