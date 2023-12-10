# coding=utf-8
# Filename  :时间合并
# Time      :2022/11/4

import copy
import gc
from osgeo import gdal
import numpy as np


# 首先来分步骤
# 1.基于每年的classify分出小space-cube
# space-cube: {id:number, coordinate:[[x,y],...], value:[v1, v2, ...]}
def calc_space_cube():
    for year in range(63):
        obj = {}
        classify = np.load(f'./change-version/test/classify-last-{year}.npy')
        value = np.load(f'./data/evi/{year}.npy')
        for i in range(classify.shape[0]):
            for j in range(classify.shape[1]):
                if not np.isnan(value[i, j]):
                    if obj.get(classify[i][j]):
                        obj[classify[i, j]]['coordinate'].append([i, j])
                        obj[classify[i, j]]['value'].append(value[i, j])
                    else:
                        obj[classify[i, j]] = {'id': classify[i, j], 'coordinate': [[i, j]], 'value': [value[i, j]]}
        np.save(f'./change-version/test/time-cube/space-cube-dict{year}.npy', obj)
        print(f'{year} was finished')


# 2.制作连接矩阵connect: [[]]
def calc_connect(year):
    last = np.load(f'./change-version/test/save-npy/st-id-{year}.npy')
    current = np.load(f'./change-version/test/classify-last-{year + 1}.npy')
    img1 = np.load(f'./change-version/test/value-{year}.npy')
    img2 = np.load(f'./change-version/test/value-{year + 1}.npy')
    # st_cube1 = np.load(f'./time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    # st_cube2 = np.load(f'./time-cube/space-cube-dict{year + 1}.npy', allow_pickle=True).item()
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
    np.save(f'./change-version/test/time-cube/connect{year}.npy', connect)
    return connect


# 将第一年的cube直接作为大的时空cube构建首个STcube，后面根据这个cube时间上生长
def build_first_cube(year):
    # year = 0
    first = np.load(f'./change-version/test/time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    for key in list(first.keys()):
        if len(first[key]['value']) <= 1:
            first.pop(key)
    obj = {}
    for item in first:
        obj[item] = [{'sci': item, 'year': year, 'value': first[item]['value'],
                      'coordinate': first[item]['coordinate']}]
    np.save(f'./change-version/test/time-cube/ST_cube_dict{year}.npy', obj)


def merge_cube(year):
    space_cube = np.load(f'./change-version/test/time-cube/space-cube-dict{year}.npy', allow_pickle=True).item()
    before = len(space_cube)
    print('before', before)
    st_cube = np.load(f'./change-version/test/time-cube/ST_cube_dict{year - 1}.npy', allow_pickle=True).item()
    connect = calc_connect(year - 1)
    print('connect', len(connect))
    classify = np.load('./change-version/test/classify-last-0.npy')
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
        if sh < 80:
            space_cube[con[1]]['year'] = year
            st_cube[con[0]].append(space_cube[con[1]])
            for cor in space_cube[con[1]]['coordinate']:
                nowid[cor[0], cor[1]] = con[0]
            space_cube.pop(con[1])
    # 最后没有判别上的cube，也就是sh>=20的cube，单独起一个cube放到st_cube里面
    print('merge', before - len(space_cube))
    for item in list(space_cube.keys()):
        if len(space_cube[item]['value']) < 2:
            space_cube.pop(item)
    print('rest', len(space_cube))
    m = max(st_cube.keys())
    for key in space_cube:
        m += 1
        st_cube[m] = [{'sci': m, 'year': year, 'value': space_cube[key]['value'],
                       'coordinate': space_cube[key]['coordinate']}]
        # space_cube.pop(key)
        for cor in space_cube[key]['coordinate']:
            nowid[cor[0], cor[1]] = m
    np.save(f'./change-version/test/time-cube/ST_cube_dict{year}.npy', st_cube)
    np.save(f'./change-version/test/save-npy/st-id-{year}.npy', nowid)
    return st_cube, nowid


def main():
    for year in range(1, 63):
        st_cube, _ = merge_cube(year)
        print(f'{year} was finished', len(st_cube))


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


def calc_stgs(ST_cube_dict, connect):
    # 先计算intra
    h_intra_dic = {}
    obj = {}
    numobj = {}
    stgs = 0
    for item in ST_cube_dict:
        valuelist = []
        for dic in ST_cube_dict[item]:
            valuelist += dic['value']
        h_intra_dic[item] = np.std(valuelist)
        obj[item] = np.mean(valuelist)
        numobj[item] = len(valuelist)
    # 计算inter
    h_inter_dic = {}
    for key in connect:
        temp = [obj[key]]
        for nb in connect[key]:
            temp.append(obj[nb])
        h_inter_dic[key] = np.std(temp)
    for key in h_inter_dic:
        h_inter = h_inter_dic[key]
        h_intra = h_intra_dic[key]
        ls = 1-h_intra/h_inter
        stgs += ls*numobj[key]
    stgs/np.sum(numobj.values())
    return stgs


def connect_array(st_id):
    connect = {}
    for i in range(st_id.shape[0]-1):
        for j in range(st_id.shape[1]-1):
            # 不是nan的情况下，判断下和右边是否有不同的编号
            # 判断下

            if st_id[i, j] != -1 and st_id[i + 1, j] != -1 and st_id[i + 1, j] != st_id[i, j]:
                if connect.get(st_id[i, j]):
                    connect[st_id[i, j]].append(st_id[i + 1, j])
                else:
                    connect[st_id[i, j]] = [st_id[i + 1, j]]
                if connect.get(st_id[i + 1, j]):
                    connect[st_id[i + 1, j]].append(st_id[i, j])
                else:
                    connect[st_id[i + 1, j]] = [st_id[i, j]]
                # 判断右
                if st_id[i, j] != -1 and st_id[i, j + 1] != -1 and st_id[i, j + 1] != st_id[i, j]:
                    if connect.get(st_id[i, j]):
                        connect[st_id[i, j]].append(st_id[i, j + 1])
                    else:
                        connect[st_id[i, j]] = [st_id[i, j + 1]]
                    if connect.get(st_id[i, j + 1]):
                        connect[st_id[i, j + 1]].append(st_id[i, j])
                    else:
                        connect[st_id[i, j + 1]] = [st_id[i, j]]
    for key in connect:
        connect[key] = list(np.unique(connect[key]))
    return connect
