# -*- coding: utf-8 -*-
# Time : 2020/1/6 0006  16:45 
# Author : dengfan
'''
bbox_head=dict(
        type='RetinaHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='GHMC',
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='GHMR', mu=0.02, bins=10, momentum=0.7, loss_weight=10.0)))
'''