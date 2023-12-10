# _*_ coding: utf-8 _*_
# H-inter H-intra STGS num
# si ti sti
# |xi-xa|/std
# 每个时刻立方体内像元个数的集合的标准差
import numpy as np
from Function_Set import read_tif, write_tif
from osgeo import gdal
gdal.GDT_Byte

def calc_stgs(ST_cube_dict, connect, yz):
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
    for item in obj:
        if not h_inter_dic.get(item):
            h_inter_dic[item] = np.std([obj[item], -1])
    # 计算stgs
    numcount = 0
    for key in h_intra_dic:
        h_intra = h_intra_dic[key]
        if h_inter_dic.get(key):
            h_inter = h_inter_dic[key]
        else:
            h_inter = np.std([obj[key], -1])
        if h_intra / h_inter < yz and key != -999:
            ls = 1 - h_intra / h_inter
            stgs += ls * numobj[key]
            numcount += numobj[key]
    stgs = stgs / numcount
    #     ls = 1 - h_intra / h_inter
    #     stgs += ls*numobj[key]
    # stgs = stgs/np.sum(list(numobj.values()))
    return np.mean(list(h_intra_dic.values())), np.mean(list(h_inter_dic.values())), stgs


def connect_array(st_id):
    connect = {}
    for i in range(st_id.shape[0] - 1):
        for j in range(st_id.shape[1] - 1):
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


def calc_si():
    for i in range(5):
        shli = [10, 20, 30, 40, 50]
        sh = shli[i]
        path = f'./{sh}'
        yzli = [20, 10, 5, 10, 20]
        for end in [20, 41, 62]:
            st_cube = np.load(f'{path}/time-cube/ST_cube_dict{end}.npy', allow_pickle=True).item()
            del st_cube[-999]
            st_id = np.load(f'{path}/save-npy/st-id-{end}.npy', allow_pickle=True)
            connect = connect_array(st_id)
            h_intra, h_inter, stgs = calc_stgs(st_cube, connect, yzli[i])
            print(sh, end, h_intra, h_inter, stgs, len(st_cube))


def calc_dg():
    for sh in [30]:
        path = f'./{sh}'
        for end in [20, 41, 62]:
            st_cube = np.load(f'{path}/time-cube/ST_cube_dict{end}.npy', allow_pickle=True).item()
            del st_cube[-999]
            st_id = np.load(f'{path}/save-npy/st-id-{end}.npy', allow_pickle=True)
            connect = connect_array(st_id)
            obj = {}
            numobj = {}
            si_dic = {}
            ti_dic = {}
            sti_li = []
            for item in st_cube:
                valuelist = []
                ti_dic[item] = len(st_cube[item])
                for dic in st_cube[item]:
                    valuelist += dic['value']
                obj[item] = np.mean(valuelist)
                numobj[item] = len(valuelist)
                sti_li.append(len(st_cube[item]))
            for key in connect:
                value = obj[key]
                templist = [value]
                for lj in connect[key]:
                    templist.append(obj[lj])
                si = abs(value - np.mean(templist)) / np.std(templist + value)
                si_dic[key] = si
            np.save(f'si-{sh}-{end}.npy')
            print(si_dic, ti_dic, np.std(sti_li))


# 计算si ti sti
# 每个立方体的光谱值减去所有立方体的光谱平均值除以所有光谱的标准差
# 每个立方体持续时间-1除以20
# 立方体个数的标准差
def calc_rest_three():
    for sh in [30]:
        path = f'./{sh}'
        for end in [20, 41, 62]:
            st_cube = np.load(f'{path}/time-cube/ST_cube_dict{end}.npy', allow_pickle=True).item()
            del st_cube[-999]
            si_dic = {}
            ti_dic = {}
            sti_dic = {}
            for key in st_cube:
                sti_arr = []
                temp = []
                # 计算ti,
                yearlist = []
                ti = 0
                ti_dic[key] = (len(st_cube[key]) - 1) / 20
                for i in range(len(st_cube[key])):
                    item = st_cube[key][i]
                    if item['year'] not in yearlist:
                        ti += 1
                        yearlist.append(item['year'])
                    temp += item['value']
                    sti_arr.append(len(item['value']))
                    # st_cube[key][i]['mv'] = np.mean(item['value'])
                ti_dic[key] = (ti - 1) / 20
                # 每个立方体的平均值
                si_dic[key] = np.mean(temp)
                # 计算sti, 修正sti去除太小的值
                sti_arr = np.array(sti_arr)
                sti_arr = sti_arr[sti_arr > np.mean(sti_arr)/2]

                sti_dic[key] = np.std(sti_arr)
            # 计算si
            x_ = np.mean(list(si_dic.values()))
            std = np.std(list(si_dic.values()))
            for j in si_dic:
                si = np.abs(si_dic[j] - x_) / std
                si_dic[j] = si
            classify = np.load(f'{path}/classify/classify-0.npy')
            res1 = [np.full(classify.shape, np.nan) for _ in range(21)]
            res2 = [np.full(classify.shape, np.nan) for _ in range(21)]
            res3 = [np.full(classify.shape, np.nan) for _ in range(21)]
            if end == 41:
                dec = 21
            elif end == 62:
                dec = 42
            elif end == 20:
                dec = 0
            for k in st_cube:
                for lft in st_cube[k]:
                    for cor in lft['coordinate']:
                        res1[lft['year'] - dec][cor[0], cor[1]] = si_dic[k]
                        res2[lft['year'] - dec][cor[0], cor[1]] = ti_dic[k]
                        res3[lft['year'] - dec][cor[0], cor[1]] = sti_dic[k]
            im_proj, im_Geotrans, im_data = read_tif('./S2_20190408(1).tif')
            for i in range(21):
                write_tif(f'./rest-3/si-{end - 20 + i}.tif', res1[i], im_Geotrans, im_proj, gdal.GDT_Float32)
                write_tif(f'./rest-3/ti-{end - 20 + i}.tif', res2[i], im_Geotrans, im_proj, gdal.GDT_Float32)
                write_tif(f'./rest-3/sti-{end - 20 + i}.tif', res3[i], im_Geotrans, im_proj, gdal.GDT_Float32)


if __name__ == '__main__':
    calc_rest_three()
