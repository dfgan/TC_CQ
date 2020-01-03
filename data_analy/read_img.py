# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  16:33 
# Author : dengfan

import os, cv2

files = os.listdir('big_img')
img = cv2.imread(os.path.join('big_img', files[0]))
print(img.shape)
print(img[0])
print('...........')
print(img[1])