# _*_ coding: utf-8 _*_
from PyEMD import CEEMDAN
import time
import numpy as np

star = time.time()
num = 10
end = 11
interlist = np.load(r"./riceInterlist.npy")
IMFS = {}
ceemdan = CEEMDAN()
for i in range(num*10000, end*10000):
    print(f'\r {i}/{110000}', end='')
    list = interlist[i][2:]
    imfs_ceemd = ceemdan(list)  # 信号分解
    IMFS[i] = imfs_ceemd

np.save(rf"./ndriIMFS{num}.npy", IMFS)

end = time.time()
print(f'process end with time {int((end - star) / 60)}')
