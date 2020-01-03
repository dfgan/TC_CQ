# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  14:43 
# Author : dengfan
import json
import os
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

    annotations = {}
    for i in data['annotations']:
        id = i['image_id']
        bbox = i['bbox']
        category = i['category_id']
        # print(id)
        # print(image_name[id])
        if category == 0:
            continue
        if image_name[id] not in annotations:
            annotations[image_name[id]] = [[bbox[0], bbox[1], bbox[2], bbox[3], label[category]], ]
        else:
            annotations[image_name[id]].append([bbox[0], bbox[1], bbox[2], bbox[3], label[category]])

    return annotations, img_info

annotations, img_info = get_annotations()

#image size info statistics
# size = {}
# recode = []
# for name in img_info:
#     dic = img_info[name]
#     w = dic['width']
#     h = dic['height']
#     if [w, h] not in recode:
#         recode.append([w, h])
#         size[str(w)+'_'+str(h)] = 1
#     else:
#         size[str(w) + '_' + str(h)] += 1
#
# print(size)

#bbox statistics
bbox_dict_smal = {}
bbox_dict_big = {}
for name in annotations:
    bbox_info = annotations[name]
    for bbox_ in bbox_info:
        w_h = [bbox_[2], bbox_[3]]
        label = bbox_[4]
        #根据图片大小来判断时大图还是小图框
        if img_info[name]['width'] > 1000:
            if label not in bbox_dict_big:
                bbox_dict_big[label] = [w_h,]
            else:
                bbox_dict_big[label].append(w_h)
        else:
            if label not in bbox_dict_smal:
                bbox_dict_smal[label] = [w_h,]
            else:
                bbox_dict_smal[label].append(w_h)

color = ['#8FBC8F', '#8B4513', '#00FFFF', '#9ACD32',
         '#D8BFD8', '#4682B4', '#4169E1', '#B0E0E6', '#FFEFD5',
         '#808000', '#F5FFFA', '#00FA9A', '#0000CD', '#FFFFE0', '#F0E68C',
         '#ADFF2F']
num =0
for label in bbox_dict_big:
    print(label)
    bboxes = np.asarray(bbox_dict_big[label])
    plt.scatter(list(bboxes[:, 0]), list(bboxes[:, 1]), color=color[num], label=label)
    num += 1
plt.legend(loc = 'upper left')
plt.show()


'''
big_map_code : BTQZ, BTQP, BTWX                                 3
small_map_code : PGBX, PGDD, PMZC, PGHB, PGPS, PMYC, PGDX       7
'''