# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  15:31 
# Author : dengfan
import json, os
import numpy as np
import copy

''''
images = [{"file_name": "cat.jpg", "id":1},
		  {"file_name": "dog.jpg", "id":2}]
annotations = [{"image_id":1, "bbox":[100.00, 200.00, 10.00, 10.00], "category_id": 1, "score":0.98},
			   {"image_id":2, "bbox":[150.00, 250.00, 20.00, 20.00], "category_id": 2, "score":0.92}]
predictions = {"images":images, "annotations":annotations}

with open("predictions.json", 'w+') as f:
	json.dump(predictions, f)

	"categories": [
		{
			"supercategory": "瓶盖破损",
			"id": 1,
			"name": "瓶盖破损"
		},
		{
			"supercategory": "喷码正常",
			"id": 9,
			"name": "喷码正常"
		},
		{
			"supercategory": "瓶盖断点",
			"id": 5,
			"name": "瓶盖断点"
		},
		{
			"supercategory": "瓶盖坏边",
			"id": 3,
			"name": "瓶盖坏边"
		},
		{
			"supercategory": "瓶盖打旋",
			"id": 4,
			"name": "瓶盖打旋"
		},
		{
			"supercategory": "背景",
			"id": 0,
			"name": "背景"
		},
		{
			"supercategory": "瓶盖变形",
			"id": 2,
			"name": "瓶盖变形"
		},
		{
			"supercategory": "标贴气泡",
			"id": 8,
			"name": "标贴气泡"
		},
		{
			"supercategory": "标贴歪斜",
			"id": 6,
			"name": "标贴歪斜"
		},
		{
			"supercategory": "喷码异常",
			"id": 10,
			"name": "喷码异常"
		},
		{
			"supercategory": "标贴起皱",
			"id": 7,
			"name": "标贴起皱"
		}
'''



def py_cpu_nms(dets, thresh):
    # dets:(m,5)  thresh:scaler

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1

    return keep

def class_change(id):
    '''
    index = ["PGBX", "PGDD", "PMZC", "PGHB", "PGPS", "PMYC", "PGDX", "BTQZ", "BTQP", "BTWX" ]
    :param id:
    :return:
    '''
    train_class = {1:2, 2:5, 3:9, 4:3, 5:1, 6:10, 7:4, 8:7, 9:8, 10:6}
    return train_class[int(id)+1]


def get_result(pre):

    f = open(pre, encoding='utf-8')
    dic = json.load(f)
    # dic = json.loads(setting)
    for i in dic:
        print(i)

    annotations = dic['annotations']
    images = dic['images']
    # print(annotations[0])

    imgs = {}           #key=index,value=name
    for img in images:
        file_name = img['file_name']
        id = img['id']
        if id in imgs:
            print(id)
            print(file_name)
            raise Exception('文件{} 已经存在，出现重复编码！！！'.format(file_name))
        imgs[id] = file_name

    result = {}
    for ann in annotations:
        img_id = ann['image_id']
        bbox = ann['bbox']
        label = class_change(ann['category_id'])
        score = ann['score']

        if imgs[img_id] not in result:
            result[imgs[img_id]] = [[bbox[0], bbox[1], bbox[2], bbox[3], score, label], ]
        else:
            result[imgs[img_id]].append([bbox[0], bbox[1], bbox[2], bbox[3], score, label])

    return imgs, result

def deal_with_bbox(info):       #bbox[:4] + score + label
    data = np.asarray(copy.deepcopy(info))
    label = data[:, 5]
    label_set = set(list(label))
    result = []
    for i in label_set:
        index = np.where(label == i)
        bbox = data[index]
        keep = py_cpu_nms(bbox[:, :5], iou_th)
        out = bbox[keep, :]
        result.extend(out)
    return result


if __name__ == "__main__":
    pre = r'predictions.json'

    iou_th = 0.5
    score_th = 0.01

    _, result = get_result(pre)

    images = []
    annotations = []
    index = 1
    recode_name = []
    for name in result:
        data = deal_with_bbox(result[name])
        anns = []

        for box in data:
            if float(box[4]) < score_th:
                continue
            ann = {"image_id":index, "bbox":list([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]), "category_id": int(box[5]), "score":float(box[4])}
            anns.append(ann)

        if name not in recode_name and len(anns) > 0:
            recode_name.append(name)
            image = {"file_name":name, "id":index}
            images.append(image)
            index += 1
        if len(anns) > 0:
            annotations.extend(anns)

    predictions = {"images": images, "annotations": annotations}

    with open("result.json", 'w+') as f:
        jsondata = json.dumps(predictions, indent=4, separators=(',', ': '))
        f.write(jsondata)
