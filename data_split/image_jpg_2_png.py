# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  17:53 
# Author : dengfan

import os, cv2

image_dir = r'E:\study_code\TC_CQ\data\train\images'
save_dir = r'E:\study_code\TC_CQ\data\train\image_png'
names = os.listdir(image_dir)

for name in names:
    path = os.path.join(image_dir, name)
    img = cv2.imread(path)
    save_path = os.path.join(save_dir, name.split('.')[0]+'.png')
    cv2.imwrite(save_path, img)