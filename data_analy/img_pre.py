# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  16:06 
# Author : dengfan
import cv2, os, shutil

img_path = r'E:\study_code\TC_CQ\data\test\images'
files = os.listdir(img_path)

for i in files:
    path = os.path.join(img_path, i)
    img = cv2.imread(path)
    if img.shape[0] < 1000:
        shutil.copyfile(path, os.path.join('small_images', i))
        print(i)