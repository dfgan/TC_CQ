# -*- coding: utf-8 -*-
# Time : 2019/12/25 0025  17:16 
# Author : dengfan
from lxml import etree
import os


def get_label(name):
    xml = etree.parse(name)  # 读取test.xml文件
    root = xml.getroot()  # 获取根节点
    label = []
    for i in root.getchildren():
        if i.tag == 'object':
            for n in i.getchildren():
                if n.tag == 'name':
                    label.append(n.text)
    return label

def get_name(name):
    names = []
    with open(name, 'r') as r:
        lines = r.readlines()
        for line in lines:
            names.append(line.strip()+'.xml')
    return names

if __name__ == '__main__':
    xml_dir = r'E:\study_code\TC_CQ\data\train\xml_new'
    train_file = r'train_small.txt'
    valid_file = r'valid_small.txt'

    train_label = []
    valid_label = []

    train_name = get_name(train_file)
    valid_name = get_name(valid_file)
    print(train_name)

    for name in train_name:
        path = os.path.join(xml_dir, name)
        labels = get_label(path)
        for label in labels:
            if label not in train_label:
                train_label.append(label)

    for name in valid_name:
        path = os.path.join(xml_dir, name)
        labels = get_label(path)
        for label in labels:
            if label not in valid_label:
                valid_label.append(label)

    print('train:')
    print(train_label)
    print('valid:')
    print(valid_label)