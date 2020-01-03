# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  11:37 
# Author : dengfan

import json
import os
import cv2

label = {1:'PGPS', 9:'PMZC', 5:'PGDD', 3:'PGHB', 4:'PGDX', 2:'PGBX', 8:'BTQP', 6:'BTWX', 10:'PMYC', 7:'BTQZ', 0:'BG'}


images_dir = r'E:\study_code\TC_CQ\data\train\images'
json_file_dir = r'E:\study_code\TC_CQ\data\train\annotations.json'
save_dir = r'E:\study_code\TC_CQ\data\train\clip_images'

with open(json_file_dir, 'r') as f:
    data = json.load(f)
for k in data:
    print(k)

image_name = {}
for i in data['images']:
    name = i['file_name']
    id = i['id']
    if id in image_name:
        print('error !!!')
        print(id)
    image_name[id] = name


annotations = {}
for i in data['annotations']:
    id = i['image_id']
    bbox = i['bbox']
    category = i['category_id']
    if image_name[id] not in annotations:
        annotations[image_name[id]] = [[bbox, label[category]],]
    else:
        annotations[image_name[id]].append([bbox, label[category]])

print(annotations)

images_files = os.listdir(images_dir)
for image in images_files:
    path = os.path.join(images_dir, image)
    infos = annotations[image]
    num = 0
    for info in infos:
        num += 1
        bbox = info[0]
        cla = info[1]
        name = image.split('.')[0]+'_'+cla+'_'+str(num)+'.jpg'
        img = cv2.imread(path)

        save_img = img[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2]),:]
        save_path = os.path.join(save_dir, cla)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, name)
        print(save_img.shape)
        if save_img.shape[0] <=0 or save_img.shape[1]<=0:
            continue
        cv2.imwrite(save_name, save_img)