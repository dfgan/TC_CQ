# -*- coding: utf-8 -*-
# Time : 2020/1/3 0003  8:59 
# Author : dengfan
import json, os, cv2
import numpy as np
import copy

from PIL import Image, ImageDraw, ImageFont

'''
 index = ["PGBX", "PGDD", "PMZC", "PGHB", "PGPS", "PMYC", "PGDX", "BTQZ", "BTQP", "BTWX" ]
'''

classes = {
    0: '背景',
	1: 'PGPS',#'瓶盖破损',
	2: 'PGBX',#'瓶盖变形',
	3: 'PGHB',#'瓶盖坏边',
	4: 'PGDX',#'瓶盖打旋',
	5: 'PGDD',#'瓶盖断点',
	6: 'BTWX',#'标贴歪斜',
	7: 'BTQZ',#'标贴起皱',
	8: 'BTQP',#'标贴气泡',
	9: 'PMZC',#'喷码正常',
	10: 'PMYC'#'喷码异常'}
}

def read_json(files_path):
    with open(files_path, encoding='utf-8') as f:
        dic = json.load(f)
        # dic = json.loads(setting)
    return dic

def get_dic(data):
    data = read_json(path)
    for i in data:
        print(i)

    images = data['images']
    data_images = {}
    for dic in images:
        file_name = dic['file_name']
        id = dic['id']
        if id in data_images:
            print(id)
            print(file_name)
            raise Exception('文件{} 已经存在，出现重复编码！！！'.format(file_name))
        data_images[id] = file_name

    annotations = data['annotations']
    out = {}
    for ann in annotations:
        name = data_images[ann['image_id']]
        bbox = ann['bbox']
        category = ann['category_id']
        score = ann['score']
        if name not in out:
            out[name] = [[bbox[0], bbox[1], bbox[2], bbox[3], category, score], ]
        else:
            out[name].append([bbox[0], bbox[1], bbox[2], bbox[3], category, score])
    return out



if __name__ == "__main__":
    path = r'../process_after/result.json'
    img_path = r'E:\study_code\TC_CQ\data\test\images'
    save_path = r'./show_imgs2'

    data = get_dic(read_json(path))

    num = 0
    for name in data:
        num += 1
        # if num > 10:
        #     break
        name_path = os.path.join(img_path, name)
        print(name_path)
        bboxes = data[name]
        img = cv2.imread(name_path)
        for bbox in bboxes:
            if bbox[5] < 0.15:
                continue
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),
                                (0, 0, 255), thickness=1)

            img = cv2.putText(img, classes[bbox[4]]+'-'+ str(bbox[5])[:6], (int(bbox[0]), int(bbox[1]) -5),
                              cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        save_name = os.path.join(save_path, name)
        cv2.imwrite(save_name, img)