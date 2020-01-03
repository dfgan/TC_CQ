# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  16:00 
# Author : dengfan
import cv2, os, json
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

big_name = []
small_name = []
for name in img_info:
    w = img_info[name]['width']
    if w > 1000:
        big_name.append(name)
    else:
        small_name.append(name)

with open('big_img_name.txt', 'w+') as w:
    for name in big_name:
        w.write(name + '\n')

with open('small_img_name.txt', 'w+') as w:
    for name in small_name:
        w.write(name+'\n')