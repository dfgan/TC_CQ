# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  11:15 
# Author : dengfan
import json
import os

json_file_dir = r'E:\study_code\TC_CQ\data\train\annotations.json'

with open(json_file_dir, 'r') as f:
    data = json.load(f)
for k in data:
    print(k)
images = data['images']
size = []
for i in images:
    w = i['width']
    h = i['height']
    if [w,h] not in size:
        size.append([w,h])
print(size)

new = {'images':data['images'], 'annotations':data['annotations'], 'categories':data['categories']}
with open('json_new.json', 'w+') as w:
    json.dump(new, w)

'''
"id": 1,"name": "瓶盖破损"

"id": 9,"name": "喷码正常"

"id": 5,"name": "瓶盖断点"

"id": 3,"name": "瓶盖坏边"

"id": 4,"name": "瓶盖打旋"

"id": 0,"name": "背景"

"id": 2,"name": "瓶盖变形"

"id": 8,"name": "标贴气泡"

"id": 6,"name": "标贴歪斜"

"id": 10,"name": "喷码异常"

"id": 7,"name": "标贴起皱"
'''