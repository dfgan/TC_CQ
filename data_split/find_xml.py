# -*- coding: utf-8 -*-
# Time : 2019/12/30 0030  16:29 
# Author : dengfan
import os, shutil

xml_dir = r'E:\study_code\TC_CQ\data\train\xml_new'
save_dir = r'E:\study_code\TC_CQ\data\train\xml_small'
train_file = r'train_small.txt'
valid_file = r'valid_small.txt'

all_xml = os.listdir(xml_dir)


def get_name(name):
    names = []
    with open(name, 'r') as r:
        lines = r.readlines()
        for line in lines:
            names.append(line.strip()+'.xml')
    return names

small = []
name_train = get_name(train_file)
small.extend(name_train)
small.extend(get_name(valid_file))

for name in small:
    shutil.copyfile(os.path.join(xml_dir, name), os.path.join(save_dir, name))
