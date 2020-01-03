# -*- coding: utf-8 -*-
# Time : 2019/12/30 0030  17:21 
# Author : dengfan
import shutil, os

xml_path = r'E:\study_code\TC_CQ\data\train\xml_new'

train_files = r'train_big.txt'
valid_files = r'valid_big.txt'

train_save = r'./big_train_xml'
valid_save = r'./big_valid_xml'

def get_name(name):
    names = []
    with open(name, 'r') as r:
        lines = r.readlines()
        for line in lines:
            names.append(line.strip()+'.xml')
    return names

train_xml = get_name(train_files)
valid_xml = get_name(valid_files)

for i in train_xml:
    shutil.copyfile(os.path.join(xml_path, i), os.path.join(train_save, i))

for i in valid_xml:
    shutil.copyfile(os.path.join(xml_path, i), os.path.join(valid_save, i))