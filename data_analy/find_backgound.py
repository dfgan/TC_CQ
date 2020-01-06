# -*- coding: utf-8 -*-
# Time : 2020/1/6 0006  17:10 
# Author : dengfan

import json
import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import cv2
from lxml import etree, objectify

label = {1: 'PGPS', 9: 'PMZC', 5: 'PGDD', 3: 'PGHB', 4: 'PGDX', 2: 'PGBX', 8: 'BTQP', 6: 'BTWX', 10: 'PMYC',
             7: 'BTQZ', 0: 'BG'}
images_dir = r'E:\study_code\TC_CQ\data\train\images'
json_file_dir = r'E:\study_code\TC_CQ\data\train\annotations.json'
save_dir = r'E:\study_code\TC_CQ\data\train\clip_images'
xml_dir = r'E:\study_code\TC_CQ\data\train\xml_new'


def get_annotations():

    with open(json_file_dir, 'r') as f:
        data = json.load(f)
    for k in data:
        print(k)

    image_name = {}
    img_info = {}
    for i in data['images']:
        name = i['file_name']
        id = i['id']
        if id in image_name:
            print('error !!!')
            print(id)
        image_name[id] = name
        img_info[name] = {'file_name':name, 'width':i['width'], 'height':i['height'], 'path':os.path.join(images_dir, name)}

    back_gound_files = {}
    for i in data['annotations']:
        id = i['image_id']
        category = i['category_id']
        if category == 0:
            if image_name[id] not in back_gound_files:
                back_gound_files[image_name[id]] = []
        else:
            if image_name[id] in back_gound_files:
                back_gound_files[image_name[id]].append(label[category])

    return back_gound_files

back_gound_files = get_annotations()
num1 = 0
num2 = 0
bg_files = []
for file in back_gound_files:
    num1 += 1
    if len(back_gound_files[file]) > 0:
        num2 += 1
        print(file)
        print(back_gound_files[file])
        print('......................')
    else:
        shutil.copyfile(os.path.join(r'E:\study_code\TC_CQ\data\train\images', file), os.path.join(r'E:\study_code\TC_CQ\data\train\bg_images', file))

print(num1)
print(num2)