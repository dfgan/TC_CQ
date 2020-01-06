# -*- coding: utf-8 -*-
# Time : 2020/1/6 0006  16:53 
# Author : dengfan

'''

sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),

iou 略好
sampler=dict(
                type='CombinedSampler',
				num=512,
				pos_fraction=0.25,
				add_gt_as_proposals=True,
				pos_sampler=dict(type='InstanceBalancedPosSampler'),
				neg_sampler=dict(
					type='IoUBalancedNegSampler',
					floor_thr=-1,
					floor_fraction=0,
					num_bins=3)),

'''