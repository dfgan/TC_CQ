# -*- coding: utf-8 -*-
# @time     :21-1-10 下午1:45

import os
import numpy as np
import random
import json

train_img_dir = '/swdata/df/cz_data/data/tile_round1_train_20201231/train_imgs'
train_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/train_annos.json'

def make_images(names, start_id):
    out = []
    label_dict = {}
    for name in names:
        start_id += 1
        mid = {}
        size = names[name]
        mid['file_name'] = name
        mid['width'] = size[0]
        mid['height'] = size[1]
        mid['id'] = start_id
        out.append(mid)
        label_dict[name] = start_id
    return out, label_dict

def make_categories(labels):
    categorys = []
    label_dict = {}
    ind = 1
    for label in labels:
        mid = {}
        mid['supercategory'] = 'none'
        mid['id'] = ind
        mid['name'] = str(label)
        label_dict[label] = ind
        ind += 1
        categorys.append(mid)

    return categorys, label_dict

def make_annotations(info, label_dict, image_dict):
    annotations = []
    index = 0
    for name in image_dict:
        data = info[name]
        for bb in data['annotations']:
            mid = {}
            mid['segmentation'] = []
            mid['area'] = 0
            mid['iscrowd'] = 0
            mid['image_id'] = image_dict[name]
            bbox_old = bb['bbox']
            bbox = []
            for i in bbox_old:
                bbox.append(int(i))
            mid['bbox'] = bbox
            mid['ignore'] = 0
            index += 1
            mid['id'] = index
            mid['category_id'] = label_dict[bb['category']]
            annotations.append(mid)
    return annotations

with open(train_json, 'r') as r:
    data_info = json.load(r)

data_dict = {}
names = {}
labels = []
for info in data_info:
    name = info['name']
    height = info['image_height']
    width = info['image_width']
    category = info['category']
    bbox = info['bbox']
    if name not in names:
        names[name] = [width, height]
    if category not in labels:
        labels.append(category)
    if name in data_dict:
        data_dict[name]['annotations'].append({'category':category, 'bbox':bbox})
    else:
        data_dict[name] = {'height':height, 'width':width, 'annotations':[{'category':category, 'bbox':bbox},]}

test_dict = {}
train_dict = {}
for name in names:
    k = random.random()
    if k > 0.9:
        test_dict[name] = names[name]
    else:
        train_dict[name] = names[name]


#train
images, images_dict = make_images(train_dict, 2020000000)
categories, labels_dict = make_categories(labels)
annotations = make_annotations(data_dict, labels_dict, images_dict)
train_out = {}
train_out['images'] = images
train_out['annotations'] = annotations
train_out['categories'] = categories
print(len(annotations))
train_out = json.dumps(train_out, ensure_ascii=False, indent=4)
with open('train.json', 'w+', encoding='utf-8') as f:
    f.write(train_out)

#test
images, images_dict = make_images(test_dict, 2021000000)
annotations = make_annotations(data_dict, labels_dict, images_dict)
test_out = {}
test_out['images'] = images
test_out['annotations'] = annotations
test_out['categories'] = categories
print(len(annotations))
test_out = json.dumps(test_out, ensure_ascii=False, indent=4)
with open('test.json', 'w+', encoding='utf-8') as f:
    f.write(test_out)
