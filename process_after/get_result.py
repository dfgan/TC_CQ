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

def bbox_together(dets, thresh_iou, thresh_score, mode='mean'):
    dets = np.array(dets)

    deal_bbox = np.where(dets[:,4] > thresh_score)[0]
    deal_bbox_leave = np.where(dets[:,4] <= thresh_score)[0]
    dets_leave = list(dets[deal_bbox_leave,:])

    mats = dets[deal_bbox, :]
    bbox_scores = mats[:, 4]

    w = mats[:, 2] - mats[:, 0]
    h = mats[:, 3] - mats[:, 1]

    areas = w * h
    order = bbox_scores.argsort()[::-1]

    new_bbox = []
    bboxes = copy.deepcopy(mats)
    need_check_index = copy.deepcopy(order)

    while need_check_index.size > 0:
        leave_max_score_index = need_check_index[0]
        bb = bboxes[leave_max_score_index]
        leave_score_index = np.delete(need_check_index, 0)
        leave_bbox = bboxes[leave_score_index]

        xx1 = np.maximum(bb[0], leave_bbox[:, 0])
        yy1 = np.maximum(bb[1], leave_bbox[:, 1])
        xx2 = np.minimum(bb[2], leave_bbox[:, 2])
        yy2 = np.minimum(bb[3], leave_bbox[:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        #
        # print(xx1)
        # print(yy1)
        # print(xx2)
        # print(yy2)
        # print(h)
        # print(w)

        inter = w * h

        # print(inter)
        # print(areas)
        # print(leave_max_score_index)
        # print(order[leave_score_index])
        # print((areas[leave_max_score_index] + areas[order[leave_score_index]] - inter))

        ovr = inter / (areas[leave_max_score_index] + areas[order[leave_score_index]] - inter)

        inds = np.where(ovr > thresh_iou)[0]
        leave_inds = np.where(ovr <= thresh_iou)[0]

        # print(ovr)
        # print(leave_inds)

        if len(inds) > 0:
            select_index = leave_score_index[inds]
            select_bbox = bboxes[select_index]

            select_bbox = np.append(select_bbox, np.asarray([bb]), axis=0)
            out_score = np.max(select_bbox[:,4])

            if mode == 'mean':
                #选择平均位置
                select_bbox[:,2:4] = select_bbox[:, 0:2] + select_bbox[:, 2:4]
                out_bb = np.mean(select_bbox,axis=0)
                # out_bb[2:4] = out_bb[2:4] - out_bb[0:2]
            else:
                #选择最小外接矩
                out_bb_xy = np.min(select_bbox[:, 0:2], axis=0)
                out_bb_wh = np.max(select_bbox[:, 2:], axis=0)
                out_bb = np.hstack((out_bb_xy, out_bb_wh))

            out_bb[4] = out_score
            new_bbox.append(out_bb)
        else:
            new_bbox.append(bb)

        need_check_index = leave_score_index[leave_inds]
    dets_leave.extend(list(new_bbox))

    return dets_leave


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
        # print(out)
        # out = bbox_together(out, 0.05, 0.15, mode='max')
        result.extend(out)
    return result


if __name__ == "__main__":
    pre = r'predictions.json'

    iou_th = 0.7
    score_th = 0.05

    _, result = get_result(pre)

    images = []
    annotations = []
    index = 1
    recode_name = []
    for name in result:
        # print(name)
        # if name != 'img_0030160.jpg':
        #     continue
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
