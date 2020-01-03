
''' Wrote by Jason Huang, 2019.07.25 '''
import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split
from xml2json import xml2json

# 0 as background [original class labels, noted labels]
classname_to_id = {"M3CanLiu": 1, "M3Esd": 2, "M3MoPo": 3, "M3Peeling": 4, "M3YiWu": 5, "M3YiWu_Open": 6, "M3YiWu_Small": 7}
# 0 as background [final class labels, merged result with original class labels above]
classname_final = {"M3CanLiu": 1, "M3Esd": 2, "M3MoPo": 3, "M3Peeling": 4, "M3YiWu": 5, "M3YiWu_Open": 6,
                   "M3YiWu_Big": 7, "M3YiWu_OK": 8, "M3YiWu_Small": 9}

labelme_path = "/data/sdv1/MyData/Data/KunShanVXN/V1/5620/"   # need end with '/
saved_coco_path = "../"  # need end with '/'
image_suffix = ".jpeg"
XML_CONVERT = True
TEST_RATIO = 0.05
MIN_SIDE = 20

SUFFIX_LEN = len(image_suffix)

def convertOriLabels2FinalLabels(srcLabel):
    if srcLabel == "M3MoPo_Small":
        return "M3MoPo"
    elif srcLabel == "M3YiWu_Light":
        return "M3YiWu_OK"
    elif srcLabel == "M2Esd":
        return "M3Esd"
    else:
        return srcLabel

    # if srcLabel == "other":
    #     return "Other"
    # else:
    #     return srcLabel


def saveClasses2Txt(classname_to_id, saved_coco_path):
    f = open(saved_coco_path + 'classes.txt', 'w')
    keyClasses_ = list(classname_to_id.keys())
    for key in keyClasses_:
        f.write(key + '\n')
    f.close()

def findClassName(json_path, classname_to_id):
    keyClasses_ = list(classname_to_id.keys())
    className_ = ''
    longestSize = 0
    for key in keyClasses_:
        if key in json_path:
            if longestSize < len(key):
                className_ = key
                longestSize = len(key)
    return className_

class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # create COCO according to json file
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            if 'object' not in obj['annotation']:
                print("Json file have not Object: ", json_path)
                continue

            # image is not exist, skip current case
            imageTemp, isExist = self._image(obj, json_path)
            if isExist:
                classNamePath = findClassName(json_path, classname_to_id)
                if classNamePath == '':
                    continue
                # imageTemp['file_name'] = classNamePath + '/' + imageTemp['file_name']
                tempFile = imageTemp['file_name']
                # print(tempFile)
                imageTemp['target_file'] = classNamePath + '/' + tempFile
                imageTemp['file_name'] \
                    = convertOriLabels2FinalLabels(classNamePath) + '/' + tempFile
                self.images.append(imageTemp)
                shapes = obj['annotation']['object']
                if isinstance(shapes, list):
                    for shape in shapes:
                        annotation = self._annotation(shape, obj['annotation']['size'])
                        self.annotations.append(annotation)
                        self.ann_id += 1
                else:
                    annotation = self._annotation(shapes, obj['annotation']['size'])
                    self.annotations.append(annotation)
                    self.ann_id += 1
                self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # create subfield 'class'of COCO
    def _init_categories(self):
        for k, v in classname_final.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # create subfield 'image'of COCO
    def _image(self, obj, path):
        isExist = True
        image = {}
        # from labelme import utils
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # print(path[0:-5])
        # print(path[0:-5] + obj['annotation']['filename'][-4:])
        postfix = obj['annotation']['filename'][-SUFFIX_LEN:]
        postfix_ = None
        if postfix.upper() == postfix:
            postfix_ = postfix.lower()
        else:
            postfix_ = postfix.upper()
        imagePath = path[0:-5] + postfix
        imagePath_ = path[0:-5] + postfix_
        if os.path.exists(imagePath):
            img_x = cv2.imread(imagePath)
            h, w = img_x.shape[:-1]
            image['height'] = h
            image['width'] = w
            image['id'] = self.img_id
            # image['file_name'] = os.path.basename(path).replace(".json", '.jpg')
            image['file_name'] = os.path.basename(path).replace(".json", postfix)
        elif os.path.exists(imagePath_):
            img_x = cv2.imread(imagePath_)
            h, w = img_x.shape[:-1]
            image['height'] = h
            image['width'] = w
            image['id'] = self.img_id
            # image['file_name'] = os.path.basename(path).replace(".json", '.jpg')
            image['file_name'] = os.path.basename(path).replace(".json", postfix_)
        else:
            print('Warning: ', imagePath, 'is missing !')
            isExist = False
        return image, isExist

    # create subfield 'annotation'of COCO
    def parse_bbox(self, points):
        bndbox = []
        # As labelImg coordinate starts with '1', not '0'
        bndbox.append(int(points['xmin']) - 1)
        bndbox.append(int(points['ymin']) - 1)
        bndbox.append(int(points['xmax']) - int(points['xmin']) + 1)
        bndbox.append(int(points['ymax']) - int(points['ymin']) + 1)
        return bndbox

    # avoid small bbox out of field(anchor cannot cover it) in training
    def fitting_bbox(self, bndbox, img_size_):
        new_bndbox = bndbox
        if bndbox[2] < MIN_SIDE:
            diff = MIN_SIDE - bndbox[2]
            offset = int((diff + 0.0) * 0.5 + 0.5)
            new_bndbox[0] -= offset
            if new_bndbox[0] < 0:
                new_bndbox[0] = 0
            new_bndbox[2] += diff
            if new_bndbox[2] >= img_size_[0]:
                new_bndbox[2] = img_size_[0] - 1
        if bndbox[3] < MIN_SIDE:
            diff = MIN_SIDE - bndbox[3]
            offset = int((diff + 0.0) * 0.5 + 0.5)
            new_bndbox[1] -= offset
            if new_bndbox[1] < 0:
                new_bndbox[1] = 0
            new_bndbox[3] += diff
            if new_bndbox[3] >= img_size_[1]:
                new_bndbox[3] = img_size_[1] - 1
        return new_bndbox

    def bbox2segmentation(self, bbox):
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        return seg

    def _annotation(self, shape, img_size):
        label = shape['name']
        label = convertOriLabels2FinalLabels(label)
        bndbox_ = self.parse_bbox(shape['bndbox'])
        img_size_ = [int(img_size['width']), int(img_size['height'])]
        bndbox = self.fitting_bbox(bndbox_, img_size_)
        annotation = { }
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_final[label])
        annotation['segmentation'] = self.bbox2segmentation(bndbox)    # [np.asarray(bndbox).flatten().tolist()]
        annotation['bbox'] = bndbox
        annotation['iscrowd'] = int(shape['difficult'])
        # True mean treat the box as background
        annotation['ignore'] = 0    # int(shape['difficult'])
        annotation['area'] = bndbox[2] * bndbox[3]
        return annotation

    # read json file，return a json object
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

if __name__ == '__main__':
    # Convert all xml to json
    if XML_CONVERT == True:
        xml2json(labelme_path)
    # Create file path
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    keyClasses_ = list(classname_final.keys())
    if not os.path.exists("%scoco/train2017/" % saved_coco_path):
        os.makedirs("%scoco/train2017" % saved_coco_path)
        for label_ in keyClasses_:
            if not os.path.exists(saved_coco_path + "coco/train2017/" + label_ + '/'):
                os.makedirs(saved_coco_path + "coco/train2017/" + label_ + '/')
    if not os.path.exists("%scoco/val2017/" % saved_coco_path):
        os.makedirs("%scoco/val2017" % saved_coco_path)
        for label_ in keyClasses_:
            if not os.path.exists(saved_coco_path + "coco/val2017/" + label_ + '/'):
                os.makedirs(saved_coco_path + "coco/val2017/" + label_ + '/')
    saveClasses2Txt(classname_final, saved_coco_path)

    train_path = []
    val_path = []
    keyClasses = list(classname_to_id.keys())
    for path_index in keyClasses:
        # obtain all json file in direction images
        json_list_path_t = glob.glob(labelme_path + path_index + "/*.json")
        # data split，split in class unitin labelme_path
        train_path_t, val_path_t = train_test_split(json_list_path_t, test_size=TEST_RATIO)
        train_path.extend(train_path_t)
        val_path.extend(val_path_t)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # convert train data to json of COCO format
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)

    # for file in train_path:
    #     shutil.copy(file.replace(".json", postfix), "%scoco/images/train2017/" % saved_coco_path)
    # for file in val_path:
    #     shutil.copy(file.replace(".json", postfix), "%scoco/images/val2017/" % saved_coco_path)

    # convert valuation data to json of COCO format
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

    trainImages = train_instance['images']
    if isinstance(trainImages, list):
        for file in trainImages:
            #shutil.copy(labelme_path + file['file_name'], "%scoco/images/train2017/" % saved_coco_path)
            file_name_ = file['target_file']
            classNamePath_ = findClassName(file['file_name'], classname_final)
            assert(classNamePath_ != '')
            shutil.copy(labelme_path + file_name_, saved_coco_path + "coco/train2017/" + classNamePath_ + '/')
    else:
        #shutil.copy(labelme_path + trainImages['file_name'], "%scoco/images/train2017/" % saved_coco_path)
        file_name_ = trainImages['target_file']
        classNamePath_ = findClassName(file['file_name'], classname_final)
        assert (classNamePath_ != '')
        shutil.copy(labelme_path + file_name_, saved_coco_path + "coco/train2017/" + classNamePath_ + '/')
    valImages = val_instance['images']
    if isinstance(valImages, list):
        for file in valImages:
            #shutil.copy(labelme_path + file['file_name'], "%scoco/images/val2017/" % saved_coco_path)
            file_name_ = file['target_file']
            classNamePath_ = findClassName(file['file_name'], classname_final)
            assert (classNamePath_ != '')
            shutil.copy(labelme_path + file_name_, saved_coco_path + "coco/val2017/" + classNamePath_ + '/')
    else:
        #shutil.copy(labelme_path + valImages['file_name'], "%scoco/images/val2017/" % saved_coco_path)
        file_name_ = valImages['target_file']
        classNamePath_ = findClassName(file['file_name'], classname_final)
        assert (classNamePath_ != '')
        shutil.copy(labelme_path + file_name_, saved_coco_path + "coco/val2017/" + classNamePath_ + '/')
