# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  16:14 
# Author : dengfan
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

file_path = r'E:\study_code\TC_CQ\data\train\images'
file_name = os.listdir(file_path)
num_files = len(file_name)

#small
num = 658 * 492 * (num_files - 411)
#big
# num = 4096 * 3000 * 411

imgs = []
R_channel = 0
G_channel = 0
B_channel = 0
for i in range(num_files):
    img = imread(os.path.join(file_path, file_name[i]))
    # print(img.shape)
    if img.shape[0] > 1000:
        continue
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
    if img.shape[0] > 1000:
        continue
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
R_G_B_mean is 155.038363, 155.038363, 155.038363
R_G_B_std is 97.321989, 97.321989, 97.321989

small_img:
R_G_B_mean is 37.375027, 49.355819, 71.854576
R_G_B_std is 49.605803, 57.121447, 79.052171
'''