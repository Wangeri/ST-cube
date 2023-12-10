# coding=utf-8
# Filename  :Natural-Breaks
# Time      :2022/11/3

import jenkspy
from osgeo import gdal
import numpy as np


def read_tif(path):
    dataset = gdal.Open(path)
    # print(dataset.GetDescription())  # 数据描述

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
    for data in range(21, 36):
        # arr = np.load(f'./save_npy/mean_value{data}.npy')
        # im_proj, im_Geotrans, arr = read_tif(f'./output_img/mean_value{data}.tif')
        im_proj, im_Geotrans, arr = read_tif(f'./output_img/mean_value{data}.tif')
        arr = arr[~np.isnan(arr)].reshape(1, -1).tolist()[0]
        breaks = jenkspy.jenks_breaks(arr, n_classes=3)
        # print(breaks)
        # for item in sta.values():
        #     arr.append(item[2])
        # arr = np.array(arr)
        # arr = arr[~np.isnan(arr)]
        # breaks = jenkspy.jenks_breaks(arr, n_classes=3)
        print(breaks[2])
    im_proj, im_Geotrans, im_data = read_tif('./output_img/mean_value0.tif')
    arr = im_data[~np.isnan(im_data)].reshape(1, -1).tolist()[0]
    breaks = jenkspy.jenks_breaks(arr, n_classes=3)
    print()
    result = np.full(im_data.shape, np.nan)
    for i in range(im_data.shape[0]):
        for j in range(im_data.shape[1]):
            if not np.isnan(im_data[i, j]):
                if im_data[i, j] < breaks[1]:
                    result[i, j] = 4
                elif breaks[1] <= im_data[i, j] < breaks[2]:
                    result[i, j] = 3
                elif im_data[i, j] >= breaks[2]:
                    result[i, j] = 2
    write_tif('./output_img/result0.tif', result, im_Geotrans, im_proj, gdal.GDT_Float32)
    print('Write an elegant code!')
