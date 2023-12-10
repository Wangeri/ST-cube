# coding=utf-8
# Filename  :Function-Set
# Time      :2022/10/28


import numpy as np
from osgeo import gdal


def save_txt(str_list: list, name):
    with open(name, 'w', encoding='utf-8') as f:
        for i in str_list:
            f.write(str(i) + '\n')


def read_tif(path):
    dataset = gdal.Open(path)
    print(dataset.GetDescription())  # 数据描述

    cols = dataset.RasterXSize  # 图像长度
    rows = dataset.RasterYSize  # 图像宽度
    im_proj = dataset.GetProjection()  # 读取投影
    im_Geotrans = dataset.GetGeoTransform()  # 读取仿射变换
    im_data = dataset.ReadAsArray(0, 0, cols, rows)  # 转为numpy格式
    # im_data[im_data > 0] = 1 #除0以外都等于1
    del dataset
    return im_proj, im_Geotrans, im_data


def write_tif(newpath, im_data, im_Geotrans, im_proj, datatype):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del new_dataset


if __name__ == "__main__":
    print('Write an elegant code!')
