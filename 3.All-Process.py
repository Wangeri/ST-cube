# _*_ coding: utf-8 _*_

from osgeo import gdal
import numpy as np
from Function_Set import write_tif, read_tif
import copy
import time
import math


# 空间合并分类
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
            # if len(y) > 60000:
            #     return False
            # else:
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
        if typ == 999:
            return
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
        self.clas[self.im == 999] = -999
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                if self.clas[i, j] == -1:
                    self.id += 1
                    self.seed_fill(i, j)
                    # self.fe.append([self.id, self.im[i, j]])
        return self.clas


# 将残差还原为n*m的遥感影像
def change_data(input_path):
    index = np.load('./20230324/riceindex.npy')
    cancha = np.load(input_path + '/EVIcancha.npy')
    for j in range(cancha.shape[1]):
        reduction = np.full((np.max(index[:, 0]) + 1, np.max(index[:, 1]) + 1), np.nan)
        for i in range(index.shape[0]):
            value = cancha[i][j]
            x = index[i][0]
            y = index[i][1]
            reduction[x][y] = value
        np.save(f'{input_path}/{j}.npy', reduction)


# 空间合并的处理
def space_merge(input_path, sh):
    for i in range(63):
        start = time.time()
        print(i, 'start', end='')
        rice = np.load(rf'{input_path}/{i}.npy')
        rice[np.isnan(rice)] = 999
        rice[rice > 0.7] = 999
        rice[rice < 0.1] = 999
        img = SpaceCube(rice, sh)
        img.classify()
        classify = img.clas
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
        np.save(f'{input_path}/mean-value/value-{i}.npy', ttt)
        np.save(f'{input_path}/classify/classify-{i}.npy', classify)
        im_proj, im_Geotrans, im_data = read_tif('S2_20190408.tif')
        write_tif(f'{input_path}/mean-value/value-{i}.tif', ttt, im_Geotrans, im_proj, gdal.GDT_Float32)
        print('\r', i, 'end by time', round((time.time() - start) / 60), np.max(classify))


# 将每一年的classify分类数据转换为dict,为后续时间合并使用
def calc_space_cube(input_path):
    for year in range(63):
        obj = {}
        classify = np.load(f'{input_path}/classify/classify-{year}.npy')
        value = np.load(f'{input_path}/{year}.npy')
        for i in range(classify.shape[0]):
            for j in range(classify.shape[1]):
                if not np.isnan(value[i, j]):
                    if obj.get(classify[i][j]):
                        obj[classify[i, j]]['coordinate'].append([i, j])
                        obj[classify[i, j]]['value'].append(value[i, j])
                    else:
                        obj[classify[i, j]] = {'id': classify[i, j], 'coordinate': [[i, j]], 'value': [value[i, j]]}
        np.save(f'{input_path}/time-cube/space-cube-dict{year}.npy', obj)
        if year in [0, 21, 42]:
            np.save(f'{input_path}/save-npy/st-id-{year}.npy', classify)
        print(f'{year} was finished')


# 制作第一年的时空立方体
def build_first_cube(year, input_path):
    first = np.load(f'{input_path}/time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    for key in list(first.keys()):
        if len(first[key]['value']) <= 1:
            first.pop(key)
    obj = {}
    for item in first:
        obj[item] = [{'sci': item, 'year': year, 'value': first[item]['value'],
                      'coordinate': first[item]['coordinate']}]
    np.save(f'{input_path}/time-cube/ST_cube_dict{year}.npy', obj)


# 计算本年和上一年的时空立方体之间的立方体联系
def calc_connect(year, path):
    last = np.load(f'{path}/save-npy/st-id-{year}.npy')
    current = np.load(f'{path}/classify/classify-{year + 1}.npy')
    img1 = np.load(f'{path}/mean-value/value-{year}.npy')
    img2 = np.load(f'{path}/mean-value/value-{year + 1}.npy')
    connect = []
    used = []
    for i in range(last.shape[0]):
        for j in range(last.shape[1]):
            i_now = last[i][j]
            i_next = current[i][j]
            # 当前年份的cube不等于nan，下一年份的cube不等于nan，并且下一年份的cube没有被用过，该连接也没有存在过
            if (not np.isnan(img1[i, j])) and (not np.isnan(img2[i, j])):
                # if st_cube1.get(i_now) and st_cube2.get(i_next):
                if i_next not in used:
                    connect.append([i_now, i_next])
                    used.append(i_next)
    np.save(f'{path}/save-npy/connect{year}.npy', connect)
    return connect


# 判断th的阈值，确定是否合并
def merge_cube(path, year, th):
    space_cube = np.load(f'{path}/time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    before = len(space_cube)
    print('before', before)
    st_cube = np.load(f'{path}/time-cube/ST_cube_dict{year - 1}.npy', allow_pickle=True).item()
    connect = calc_connect(year - 1, path)
    print('connect', len(connect))
    classify = np.load(f'{path}/classify/classify-0.npy')
    nowid = np.full(classify.shape, -1)
    del classify
    for con in connect:
        # 遍历connect函数，其中的0号位为时空的立方体编码，1号位为这一年的空间立方体，现在通过这两个index查询他们的cube属性，通过合并value
        # 两个value合并后的数值，是否能够满足sh<20，可以的话，在time-cube上添加这个small-cube，id不变，这一年的id变为时空立方体的id
        if not (st_cube.get(con[0]) and space_cube.get(con[1])):
            continue
        temp = copy.deepcopy(space_cube[con[1]]['value'])
        cubelist = st_cube[con[0]]
        for c in cubelist:
            temp.extend(c['value'])
        yzx = np.std(np.array(temp)) * len(temp) * 1
        if yzx < th:
            space_cube[con[1]]['year'] = year
            st_cube[con[0]].append(space_cube[con[1]])
            for cor in space_cube[con[1]]['coordinate']:
                nowid[cor[0], cor[1]] = con[0]
            space_cube.pop(con[1])
    # 最后没有判别上的cube，也就是th>=阈值的cube，单独起一个cube放到st_cube里面
    print('merge', before - len(space_cube))
    for item in list(space_cube.keys()):
        if len(space_cube[item]['value']) < 2:
            space_cube.pop(item)
    # 没有合并上的cube
    print('rest', len(space_cube))
    m = max(st_cube.keys())
    # 没有合并上的立方体再单独放到st-cube中作为新的立方体继续使用
    for key in space_cube:
        m += 1
        st_cube[m] = [{'sci': m, 'year': year, 'value': space_cube[key]['value'],
                       'coordinate': space_cube[key]['coordinate']}]
        # space_cube.pop(key)
        for cor in space_cube[key]['coordinate']:
            nowid[cor[0], cor[1]] = m
    np.save(f'{path}/time-cube/ST_cube_dict{year}.npy', st_cube)
    np.save(f'{path}/save-npy/st-id-{year}.npy', nowid)
    return st_cube, nowid


# 计算指数
def calc_res_index(ST_cube_dict):
    count = 0
    num = 0
    yearsum = 0
    for item in ST_cube_dict:
        for dic in ST_cube_dict[item]:
            count += len(dic['value'])
        yearsum += len(ST_cube_dict[item])
        num += 1
    # 可以计算time-avg和area-avg，下面计算stgs
    return count, num, yearsum


def console():
    input_path = '20230325'
    for sh in [5]:
        start = time.time()
        rice = np.load(rf'{input_path}/0.npy')
        img = SpaceCube(rice, sh)
        img.classify()
        classify = img.clas
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
        np.save(f'{input_path}/sh5/value-{0}.npy', ttt)
        np.save(f'{input_path}/sh5/classify-{0}.npy', classify)
        im_proj, im_Geotrans, im_data = read_tif('./20230325/S2_20190408.tif')
        write_tif(f'{input_path}/sh5/value-{sh}.tif', ttt, im_Geotrans, im_proj, gdal.GDT_Float32)
        print('\r', sh, 'end by time', round((time.time() - start) / 60))


if __name__ == '__main__':
    # 定义文件夹
    path = './20230417'
    sh = 5
    th = 100
    # change_data(path)
    console()
    # # 处理数据
    # change_data(path)
    # 空间合并
    space_merge(path, sh)
    # # 开始处理时间合并
    # # calc_space_cube(path)
    # for i in [0, 21, 42]:
    #     build_first_cube(i, path)
    #     for year in range(i + 1, i + 21):
    #         st_cube, _ = merge_cube(path, year, th)
    #     ST_cube = np.load(f'{path}/time-cube/ST_cube_dict{i + 20}.npy', allow_pickle=True).item()
    #     count, num, yearsum = calc_res_index(ST_cube)
    #     time_avg = yearsum / num
    #     area_avg = count / num
    #     print(time_avg, area_avg, num, count)
