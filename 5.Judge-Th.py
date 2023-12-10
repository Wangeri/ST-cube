# coding=utf-8
# Filename  :时间合并
# Time      :2022/11/4

import copy

from osgeo import gdal
import numpy as np


# 首先来分步骤
# 1.基于每年的classify分出小space-cube
# space-cube: {id:number, coordinate:[[x,y],...], value:[v1, v2, ...]}
def calc_space_cube():
    for year in range(59, 84):
        obj = {}
        classify = np.load(f'./save_npy/classify{year}.npy')
        value = np.load(f'./data/{year}.npy')
        for i in range(classify.shape[0]):
            for j in range(classify.shape[1]):
                if not np.isnan(value[i, j]):
                    if obj.get(classify[i][j]):
                        obj[classify[i, j]]['coordinate'].append([i, j])
                        obj[classify[i, j]]['value'].append(value[i, j])
                    else:
                        obj[classify[i, j]] = {'id': classify[i, j], 'coordinate': [[i, j]], 'value': [value[i, j]]}
        np.save(f'./time-cube/space-cube-dict{year}.npy', obj)
        print(f'{year} was finished')


# 2.制作连接矩阵connect: [[]]
def calc_connect(year):
    # now = np.load(f'./save_npy/st_id{year}.npy')
    # last = np.load(f'./save_npy/classify{year - 1}.npy')
    # img1 = np.load(f'./data/{year}.npy')
    # img2 = np.load(f'./data/{year + 1}.npy')
    # arr = []
    # used = []
    # for i in range(now.shape[0]):
    #     for j in range(now.shape[1]):
    #         if (not np.isnan(img1[i, j])) and (not np.isnan(img2[i, j])):
    #             if (now[i, j] not in used) and ([last[i, j], now[i, j]] not in arr):
    #                 arr.append([last[i, j], now[i, j]])
    # np.save(f'./time-cube/connect{year}.npy', arr)
    # return arr
    last = np.load(f'./save_npy/st_id{year}.npy')
    current = np.load(f'./save_npy/classify{year + 1}.npy')
    img1 = np.load(f'./data/{year}.npy')
    img2 = np.load(f'./data/{year + 1}.npy')
    st_cube1 = np.load(f'./time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    st_cube2 = np.load(f'./time-cube/space-cube-dict{year + 1}.npy', allow_pickle=True).item()
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
                    # if (len(st_cube1[i_now]['value']) > 1) and (len(st_cube2[i_next]['value']) > 1):
                    connect.append([i_now, i_next])
                    used.append(i_next)
    # connect = list(set(connect))
    np.save(f'./time-cube/connect{year}.npy', connect)
    return connect


# 将第一年的cube直接作为大的时空cube构建首个STcube，后面根据这个cube时间上生长
def build_first_cube(year):
    first = np.load(f'./time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    for key in list(first.keys()):
        if len(first[key]['value']) <= 1:
            first.pop(key)
    obj = {}
    for item in first:
        obj[item] = [{'sci': item, 'year': year, 'value': first[item]['value'],
                      'coordinate': first[item]['coordinate']}]
    np.save(f'./time-cube/ST_cube_dict{year}.npy', obj)


def merge_cube(year, th):
    space_cube = np.load(f'./time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    st_cube = np.load(f'./time-cube/ST_cube_dict{year - 1}.npy', allow_pickle=True).item()
    connect = calc_connect(year - 1)
    classify = np.load('./save_npy/classify0.npy')
    nowid = np.full(classify.shape, -1)
    del classify
    for con in connect:
        # 遍历connect函数，其中的0号位为时空的立方体编码，1号位为这一年的空间立方体，现在通过这两个index查询他们的cube属性，通过合并value
        # 两个value合并后的数值，是否能够满足sh<20，可以的话，在time-cube上添加这个small-cube，id不变，这一年的id变为时空立方体的id
        if not st_cube.get(con[0]) and space_cube.get(con[1]):
            continue
        temp = copy.deepcopy(space_cube[con[1]]['value'])
        cubelist = st_cube[con[0]]
        for c in cubelist:
            temp.extend(c['value'])
        sh = np.std(np.array(temp)) * len(temp) * 1
        if sh < th:
            space_cube[con[1]]['year'] = year
            st_cube[con[0]].append(space_cube[con[1]])
            for cor in space_cube[con[1]]['coordinate']:
                nowid[cor[0], cor[1]] = con[0]
            space_cube.pop(con[1])
    # 最后没有判别上的cube，也就是sh>=20的cube，单独起一个cube放到st_cube里面
    for item in list(space_cube.keys()):
        if len(space_cube[item]['value']) < 2:
            space_cube.pop(item)
    m = max(st_cube.keys())
    for key in space_cube:
        m += 1
        st_cube[m] = [{'sci': m, 'year': year, 'value': space_cube[key]['value'],
                       'coordinate': space_cube[key]['coordinate']}]
        for cor in space_cube[key]['coordinate']:
            nowid[cor[0], cor[1]] = m
    np.save(f'./time-cube/ST_cube_dict{year}.npy', st_cube)
    np.save(f'./save_npy/st_id{year}.npy', nowid)
    return st_cube, nowid


def main():
    for year in range(1, 21):
        st_cube, _ = merge_cube(year)
        print(f'{year} was finished', len(st_cube))


def calc_stgs(st_cube):
    arr = []
    for item in st_cube:
        valuelist = []
        cube = st_cube[item]
        for i in cube:
            valuelist.extend(i['value'])
        if len(valuelist) > 12:
            arr.append([np.std(np.array(valuelist)), np.mean(valuelist), len(valuelist)])
    astd = np.std(np.array(arr).transpose()[1])
    count = np.sum(np.array(arr).transpose()[2])
    mc = np.mean(np.array(arr).transpose()[2])
    stgs = 0
    for i in arr:
        stgs += (1 - i[0] / astd) * i[2]
    return stgs / count, mc


if __name__ == '__main__':
    # calc_space_cube()
    # build_first_cube(42)
    # arr = []
    # for th in [20, 30, 40, 50, 60, 70]:
    #     plist = []
    #     for year in range(43, 63):
    #         st_cube, _ = merge_cube(year, th)
    #         plist.append(len(st_cube))
    #         # print('year=', year, 'finished', len(st_cube))
    #     stgs, mc = calc_stgs(st_cube)
    #     arr.append([th, stgs, mc])
    #     print(th, stgs, mc, plist)
    # np.save('./arr.npy', arr)
    # print(arr)
    st_cube = np.load('./time-cube/ST_cube_dict83.npy', allow_pickle=True).item()
    stgs, mc = calc_stgs(st_cube)
    # print(stgs, mc)
