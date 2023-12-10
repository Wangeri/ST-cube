# coding=utf-8
# Filename  :check
# Time      :2022/11/9


import numpy as np
from Function_Set import read_tif, write_tif
from osgeo import gdal

if __name__ == "__main__":
    # st_cube = np.load('./data/original/time-cube/ST_cube_dict42.npy', allow_pickle=True).item()
    original = np.load('./data/original/0.npy')
    for i in range(21, 42):
        value = np.full(original.shape, np.nan)
        for j in range(21, 42):
            st_cube = np.load(f'./data/original/time-cube/ST_cube_dict{j}.npy', allow_pickle=True).item()
            for key in st_cube:
                cube = st_cube[key]
                for dic in cube:
                    if dic['year'] == i:
                        # 再次遍历cube计算空间平均值
                        v = []
                        for k in cube:
                          v += k['value']
                        val = np.mean(v)
                        for cor in dic['coordinate']:
                            value[cor[0], cor[1]] = val
        im_proj, im_Geotrans, im_data = read_tif('./S2_20190408.tif')
        # value[value > 0.7] = np.nan
        # value[value < 0.1] = np.nan
        write_tif(f'./data/original/img/{i}-st-mean-check3.tif', value, im_Geotrans, im_proj, gdal.GDT_Float32)
        print(i, 'was finished')
    print('Write an elegant code!')
