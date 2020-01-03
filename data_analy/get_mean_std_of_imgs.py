# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  16:14 
# Author : dengfan
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

file_path = r'E:\study_code\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
file_name = os.listdir(file_path)
num_files = len(file_name)

num = 4096 * 3000 * num_files

imgs = []
R_channel = 0
G_channel = 0
B_channel = 0
for i in range(num_files):
    img = imread(os.path.join(file_path, file_name[i]))
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for i in range(num_files):
    img = imread(os.path.join(file_path, file_name[i]))
    R_channel = R_channel + np.sum(np.power(img[:, :, 0] - R_mean, 2))
    G_channel = G_channel + np.sum(np.power(img[:, :, 1] - G_mean, 2))
    B_channel = B_channel + np.sum(np.power(img[:, :, 2] - B_mean, 2))

R_std = np.sqrt(R_channel / num)
G_std = np.sqrt(G_channel / num)
B_std = np.sqrt(B_channel / num)

# R:65.045966   G:70.3931815    B:78.0636285
print("R_G_B_mean is %f, %f, %f" % (R_mean, G_mean, B_mean))
print("R_G_B_std is %f, %f, %f" % (R_std, G_std, B_std))


'''
big_img:
R_G_B_mean is 156.639888, 156.639888, 156.639888
R_G_B_std is 97.223122, 97.223122, 97.223122

small_img:
R_G_B_mean is 0.983618, 1.301052, 1.896561
R_G_B_std is 9.994623, 12.145212, 17.190142
'''