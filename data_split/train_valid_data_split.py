# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  14:42 
# Author : dengfan
import os, random

'''
('PGPS', 'PMZC', 'PGDD', 'PGHB', 'PGDX', 'PGBX', 'BTQP', 'BTWX', 'PMYC', 'BTQZ', 'BG')
'''
xml_path = r'E:\study_code\TC_CQ\data\train\xml_new'
big_img_file = r'big_img_name.txt'
samll_img_file = r'small_img_name.txt'
big_name = []
small_name = []
with open(big_img_file, 'r') as r:
    lines = r.readlines()
    for line in lines:
        big_name.append(line.strip())

with open(samll_img_file, 'r') as r:
    lines = r.readlines()
    for line in lines:
        small_name.append(line.strip())

xml_name = [i.split('.')[0]+'.jpg' for i in os.listdir(xml_path)]

random.shuffle(big_name)
print(big_name)
big_train_names = big_name[:int(len(big_name)*0.9)]
big_valid_name = big_name[int(len(big_name)*0.9):]

with open('train_big.txt', 'w+') as w :
    for name in big_train_names:
        if name in xml_name:
            w.write(name.split('.')[0]+'\n')

with open('valid_big.txt', 'w+') as w :
    for name in big_valid_name:
        if name in xml_name:
            w.write(name.split('.')[0]+'\n')


random.shuffle(small_name)
print(small_name)
small_train_names = small_name[:int(len(small_name)*0.9)]
small_valid_name = small_name[int(len(small_name)*0.9):]

with open('train_small.txt', 'w+') as w :
    for name in small_train_names:
        if name in xml_name:
            w.write(name.split('.')[0]+'\n')

with open('valid_small.txt', 'w+') as w :
    for name in small_valid_name:
        if name in xml_name:
            w.write(name.split('.')[0]+'\n')