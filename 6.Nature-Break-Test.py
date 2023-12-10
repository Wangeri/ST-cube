# coding=utf-8
# Filename  :new_task
# Time      :2022/11/12


import jenkspy
import numpy as np
from osgeo import gdal
from Function_Set import write_tif, read_tif, save_txt


def nature_break(i):
    data = np.load(f'./time-cube/space-cube-dict{i}.npy', allow_pickle=True).item()
    arr = []
    for key in data:
        if len(data[key]['value']) > 3:
            mean = np.mean(data[key]['value'])
            if 0 < mean < 1:
                arr.append(mean)
    breaks = jenkspy.jenks_breaks(arr, n_classes=2)
    # print(i, breaks[1], breaks[2])
    save_txt(arr, f'./xiangxiantu/{i}自然断点值.txt')
    im_proj, im_Geotrans, im_data = read_tif('S2_20190408.tif')
    result = np.full(im_data.shape, np.nan)
    for key in data:
        if len(data[key]['value']) > 3:
            mean = np.mean(data[key]['value'])
            if mean < breaks[1]:
                for index in data[key]['coordinate']:
                    result[index[0], index[1]] = 1
            else:
                for index in data[key]['coordinate']:
                    result[index[0], index[1]] = 2
    d = np.load(f'./data/{i}.npy')
    for x in range(d.shape[0]):
        for y in range(d.shape[1]):
            if not 0 < d[x, y] < 1:
                d[x, y] = np.nan
    write_tif(f'./xiangxiantu/{i}原始值.tif', d, im_Geotrans, im_proj, gdal.GDT_Float32)
    write_tif(f'./xiangxiantu/{i}时刻自然断点.tif', result, im_Geotrans, im_proj, gdal.GDT_Float32)
    # arr = data[~np.isnan(data)].reshape(1, -1).tolist()[0]
    # arr = list(filter(lambda x: 1 > x > 0, arr))
    # # arr = np.array(arr)
    # # arr = arr[0<arr<1]
    # breaks = jenkspy.jenks_breaks(arr, n_classes=2)
    # # data = np.load('./data/20.npy')
    # # result =
    # for i in range(result.shape[0]):
    #     for j in range(result.shape[1]):
    #         if not np.isnan(data[i, j]):
    #             if breaks[1] > data[i, j]:
    #                 result[i, j] = 1
    #             else:
    #                 result[i, j] = 2
    # write_tif('./0时刻自然断点.tif', result, im_Geotrans, im_proj, gdal.GDT_Float32)
    # print(breaks[1])


def nature_break2():
    st_cube = np.load('./time-cube/ST_cube_dict20.npy', allow_pickle=True).item()
    arr = []
    for item in st_cube:
        temp = []
        for data in st_cube[item]:
            temp.extend(data['value'])
        arr.append(np.mean(temp))
    breaks = jenkspy.jenks_breaks(arr, n_classes=3)
    print(breaks)


if __name__ == "__main__":
    for i in range(42, 63):
        nature_break(i)
    # nature_break2()
    print('Write an elegant code!')
