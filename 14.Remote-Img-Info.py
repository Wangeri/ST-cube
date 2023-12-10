import math

from osgeo import gdal
import numpy as np
import sys
import os
import csv
import pandas as pd


def read_tiff(input_path):
    dataset = gdal.Open(input_path)
    print('处理图像波段数总共有：', dataset.RasterCount)
    # 判断是否读取到数据
    if dataset is None:
        print(f'Unable to open {input_path}.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    geo_trans = dataset.GetGeoTransform()  # 几何信息
    im_bands = dataset.RasterCount  # 波段数
    img_array = dataset.ReadAsArray()
    print('投影：', projection)
    print('几何信息：', geo_trans)
    print('波段数：', im_bands)
    return img_array


def get_seq(datalist):
    arr = []
    data = datalist[0]
    for i in range(len(data)):
        for j in range(len(data[0])):
            arr.append([i, j])
    # col = len(datalist[0])
    # row = len(datalist[0][0])
    dep = len(datalist)
    print('生成序列中')
    for index, value in enumerate(arr):
        for z in range(dep):
            arr[index].append(datalist[z][value[0]][value[1]])
    return arr


def remove_nan(datalist):
    temp = []
    for data in data_list:
        if math.nan not in data:
            temp.append(data)
    return temp


def output_csv(out_path):
    pass


if __name__ == '__main__':
    path = r'D:\sss'
    files = os.listdir(path)
    data_list = []
    for file in files:
        img_matrix = read_tiff(path + '\\' + file)
        data_list.append(img_matrix.tolist())

    seq_list = get_seq(data_list)
    writer = pd.ExcelWriter('test.xlsx')
    test = pd.DataFrame(seq_list)
    test.to_excel(writer, 'sheet', index=False, header=None)
    writer.save()
    writer.close()
    print('end')
