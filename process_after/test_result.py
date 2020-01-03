# -*- coding: utf-8 -*-
# Time : 2020/1/2 0002  14:32 
# Author : dengfan

import argparse

import cv2, os
import torch
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result

def selectClsScoreBoxFromResult(result, cls_names, thd_ng):
    assert isinstance(cls_names, (tuple, list))

    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    selectedCls = []
    selectedScore = []
    selectedBox = []
    assert(len(labels) == len(bboxes))
    '''
    for i in range(0, len(labels)):
        if (cls_names[labels[i]] == 'NG' and bboxes[i][-1] > thd_ng) or cls_names[labels[i]] == 'OK':
            selectedCls.append(cls_names[labels[i]])
            selectedScore.append(bboxes[i][-1])
            tempBox = []
            tempBox = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
            selectedBox.append(tempBox)
    '''
    return selectedCls, selectedScore, selectedBox

def show_img(result, img, save_path):
    for i in range(len(result)):
        bbox = result[i][:4]
        label = result[i][-1]
        score = result[i][4]
        # extend_size = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.1 * 0.5 + 0.5)
        # left_top_x = bbox[0] - extend_size
        # if left_top_x < 0:
        #     left_top_x = 0
        # left_top_y = bbox[1] - extend_size
        # if left_top_y < 0:
        #     left_top_y = 0
        # cv.rectangle(img_np, (left_top_x, left_top_y), (bbox[2] + extend_size, bbox[3] + extend_size),
        #              (0, 0, 255), thickness=1)
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[4])), (0, 0, 255), thickness=1)
        # strText = str(selectedCls[i])
        cv2.putText(img, str(label)+'_'+str(score), (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.imwrite(save_path, img)

def result_clear(result):
    out = []
    for i, k in enumerate(result):
        if len(k) == 0:
            continue
        for boxes in k:
            if boxes[4] < 0.1:
                continue
            mid = list(boxes)
            mid.append(i)
            out.append(i)
    return np.asarray(out)

if __name__ == '__main__':
    config_file = r'./small_cascade_rcnn_x101.py'
    checkpoint_file = r'./work_dirs/small_cascade_rcnn_x101_64x4d_fpn_1x/epoch_12.pth'
    pathimg = r'/data/sdv1/datasets/dockerinfo/data/2019_12_25_voc/test_images/'
    pathSave = r'/data/sdv1/datasets/dockerinfo/data/2019_12_25_voc/show_image/'
    # init model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    imgs = [i for i in os.listdir(pathimg)]
    # logging for saving test result

    labels = {}
    progress_bar = 0
    total_bar = len(imgs)
    for image in imgs[:10]:
        img_path = os.path.join(pathimg, image)

        img_pic = cv2.imread(img_path)
        w, h = img_pic.shape[:2]
        if w > 1000:
            continue
        result = inference_detector(model, img_pic)
        result = result_clear(result)
        print(result.shape)
        #####################################
        save_path = os.path.join(pathSave, image)
        show_img(result, img_pic, save_path)
        print(image)
        print(result)