# -*- coding: utf-8 -*-
# Time : 2019/12/31 0031  11:31
# Author : dengfan

import numpy as np
import copy

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

def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    ax.plot([x1, x2], [y1, y1], c)
    ax.plot([x1, x1], [y1, y2], c)
    ax.plot([x1, x2], [y2, y2], c)
    ax.plot([x2, x2], [y1, y2], c)
    plt.title("after nms")

def bbox_together(dets, thresh_iou, thresh_score, mode='mean'):
    dets = np.array(dets)

    deal_bbox = np.where(dets[:,4] > thresh_score)[0]
    deal_bbox_leave = np.where(dets[:,4] <= thresh_score)[0]
    dets_leave = list(dets[deal_bbox_leave,:])

    mats = dets[deal_bbox, :]
    bbox_scores = mats[:, 4]

    w = mats[:, 2]
    h = mats[:, 3]

    areas = (w+1) *(h+1)
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
        xx2 = np.minimum(bb[0]+bb[2], leave_bbox[:, 0] + leave_bbox[:, 2])
        yy2 = np.minimum(bb[1]+bb[3], leave_bbox[:, 1] + leave_bbox[:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[leave_max_score_index] + areas[order[leave_score_index]] - inter)

        inds = np.where(ovr > thresh_iou)[0]
        leave_inds = np.where(ovr <= thresh_iou)[0]


        if len(inds) > 0:
            select_index = leave_score_index[inds]
            select_bbox = bboxes[select_index]

            select_bbox = np.append(select_bbox, np.asarray([bb]), axis=0)
            out_score = np.max(select_bbox[:,4])

            if mode == 'mean':
                #选择平均位置
                select_bbox[:,2:4] = select_bbox[:, 0:2] + select_bbox[:, 2:4]
                out_bb = np.mean(select_bbox,axis=0)
                out_bb[2:4] = out_bb[2:4] - out_bb[0:2]
            else:
                #选择最小外接矩
                out_bb_xy = np.min(select_bbox[:, 0:2], axis=0)
                out_bb_wh = np.max(select_bbox[:, 2:5], axis=0)
                out_bb = np.hstack((out_bb_xy, out_bb_wh))

            out_bb[4] = out_score
            new_bbox.append(out_bb)
        else:
            new_bbox.append(bb)

        need_check_index = leave_score_index[leave_inds]
    dets_leave.extend(new_bbox)

    return np.asarray(dets_leave)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)

    boxes = np.array([[50, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [170, 220, 320, 330, 0.62],
                      [120, 120, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [200, 230, 315, 340, 0.5]])

    plot_bbox(boxes, 'b')  # before nms

    keep = py_cpu_nms(boxes, thresh=0.2)
    out = bbox_together(boxes[keep], 0.2, 0.15, mode='max')
    plot_bbox(out, 'r')  # after nms
    plot_bbox(boxes[keep])
    ax.invert_yaxis()
    plt.show()

