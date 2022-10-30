# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 22:08
# @Author  : Dreamstar
# @File    : read_npy.py
# @Desc    : 添加一个对npy的代码，方便看 metrics 和其他内容

import numpy as np

file = np.load('metrics.npy')
# file = np.load('real_prediction.npy')
print(['mae', 'mse', 'rmse', 'mape', 'mspe'])  # [ 7.373819 64.90328   8.056257       inf       inf]
print(file)  # （1，48,1）
# np.savetxt('metrics.txt',file)
