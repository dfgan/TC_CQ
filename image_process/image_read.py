# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  12:47 
# Author : dengfan
import cv2

imag_file = r'img_0000067.jpg'
img = cv2.imread(imag_file)
print(img.shape)