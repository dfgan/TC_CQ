# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  16:31
# Author : dengfan

import os
from mmdet.apis import init_detector, inference_detector, show_result
import time
import random
import numpy as np
import cv2, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (218, 227, 218)


def detect_image(model, img):
    result = inference_detector(model, img)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    # (xmin, ymin, xmax, ymax, score)
    bboxes = np.vstack(result)
    time_end = time.time()
    return (result, bboxes, labels)


def inner_nms(dets, thresh=0.2):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / areas[i]
        reverse = inter / areas[order[1:]]
        ovr = np.where(ovr > reverse, ovr, reverse)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def original_nms(dets, thresh=0.8):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def preprocess_boxes(boxes):
    if boxes is None:
        return None
    filted_indices = np.where(boxes[:, 4] >= 0.05)
    if len(filted_indices[0]) == 0:
        return None
    boxes = boxes[filted_indices]
    boxes = original_nms(boxes, thresh=0.7)
    # boxes = inner_nms(boxes, thresh=0.2)
    return boxes


def vis_bbox(img, bbox, thick=1, color=RED):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img


def vis_class(img, pos, class_str, font_scale=0.35, color=RED):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, GRAY, lineType=cv2.LINE_AA)
    return img


def save_result_image_v2(im, boxes, object_classes, savepath, path):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    tickness = 4
    im_h, im_w = im.shape[:2]

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    tickness = 2

    if boxes is not None:
        for item in boxes:
            x_min, y_min, x_max, y_max, score, label = item
            x_min, y_min, x_max, y_max = (
                int(x_min), int(y_min), int(x_max), int(y_max))
            label = object_classes[int(label)]
            label_score = ' '.join((label, str(score)[:4]))
            vis_bbox(im, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1), thick=2)
            vis_class(im, (x_min, y_min), label_score, font_scale=font_scale)

    cv2.imwrite(os.path.join(savepath, path), im)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == '__main__':
    config_file1 = r'./small_cascade_dcn_50_ga.py'
    checkpoint_file1 = r'./work_dirs/cascade_rcnn_dconv_r50_1x/epoch_24.pth'
    config_file2 = r'./big_cascade_dcn_loc.py'
    checkpoint_file2 = r'./work_dirs/big_cascade_dcn_50/epoch_36.pth'

    img_path = r'./test_images'

    model1 = init_detector(config_file1, checkpoint_file1, device='cuda:0')
    model2 = init_detector(config_file2, checkpoint_file2, device='cuda:0')
    imgs = os.listdir(img_path)

    images = []
    result = []
    img_num = 0
    for name in imgs:
        path_img = os.path.join(img_path, name)
        img_pic = cv2.imread(path_img)

        w, h = img_pic.shape[:2]
        if w < 1000:
            _, bboxes, labels = detect_image(model1, path_img)
            labels = labels.reshape(-1, 1)
        else:
            _, bboxes, labels = detect_image(model2, path_img)
            labels = labels.reshape(-1, 1) + 7
        # labels = labels.reshape(-1, 1)
        print(labels)
        labels = labels.astype(bboxes.dtype)
        boxes = np.hstack((bboxes, labels))

        if len(boxes) == 0:
            boxes = None
        else:
            boxes = preprocess_boxes(boxes)

        print(name)
        print(boxes)
        if boxes is not None:
            img_num += 1
            images.append({"file_name": name, "id": img_num})
            for box in boxes:
                result.append({"image_id": img_num, "bbox": [box[0], box[1], box[2], box[3]], "category_id": box[5],
                               "score": box[4]})
        else:
            continue
    predictions = {"images": images, "annotations": result}
    '''
    dumped = json.dumps(predictions, cls=MyEncoder)
    with open("predictions.json", 'w+') as f:
	      json.dump(dumped,f)
    '''
    with open("predictions.json", 'w+') as f:
        jsondata = json.dumps(predictions, cls=MyEncoder, indent=4, separators=(',', ': '))
        f.write(jsondata)
    print('[INFO]Done.')


