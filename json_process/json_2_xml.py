# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  13:54 
# Author : dengfan
import json
import os
import random as rnd
import cv2
from lxml import etree, objectify

label = {1: 'PGPS', 9: 'PMZC', 5: 'PGDD', 3: 'PGHB', 4: 'PGDX', 2: 'PGBX', 8: 'BTQP', 6: 'BTWX', 10: 'PMYC',
             7: 'BTQZ', 0: 'BG'}
images_dir = r'E:\study_code\TC_CQ\data\train\images'
json_file_dir = r'E:\study_code\TC_CQ\data\train\annotations.json'
save_dir = r'E:\study_code\TC_CQ\data\train\clip_images'
xml_dir = r'E:\study_code\TC_CQ\data\train\xml_new'

def make_base_tree(info):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2012'),
        E.filename(info['file_name']),
        E.path(info['path']),
        E.source(
            E.database('CQ'),
            E.annotation('pascal Voc'),
        ),
        E.size(
            E.width(info['width']),
            E.height(info['height']),
            E.depth(3)
        ),
        E.segmented(0)
    )
    return anno_tree

def object(pos):
    E = objectify.ElementMaker(annotate=False)
    object = E.object(
            E.name(pos[4]),
            E.pose('right'),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(int(pos[0])),
                E.ymin(int(pos[1])),
                E.xmax(int(pos[0]+pos[2])),
                E.ymax(int(pos[1]+pos[3]))
                    )
            )
    return object


def get_xml(image_info, poses, save_name):
    base_tree = make_base_tree(image_info)
    for pos in poses:
        object_ = object(pos)
        base_tree.append(object_)
    etree.ElementTree(base_tree).write(save_name, pretty_print=True)

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

if __name__ == '__main__':
    annotations, img_info = get_annotations()

    for name in annotations:
        save_name = os.path.join(xml_dir, name.split('.')[0]+'.xml')
        get_xml(img_info[name], annotations[name], save_name)