from mmdet.apis import init_detector, inference_detector, show_result
import os
import numpy as np
from PIL import Image
import PIL.Image as Image
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config_file = '/home/shuzhilian/Jason/mmdetection/configs/faster_rcnn_r101_fpn_1x.py'
checkpoint_file = '/home/shuzhilian/Jason/mmdetection/work_dirs/faster_rcnn_r101_fpn_1x/epoch_102.pth'

IMAGES_FORMAT = ['.jpg', '.JPG', '.bmp', '.BMP']  # 图片格式
ROW_HEIGHT = 1050   # rowheight
COL_WIDTH = 950     # colwidth
IMAGE_ROW = 6  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 4  # 图片间隔，也就是合并成一张图后，一共有几列
ADD_BORDER_H = 360
ADD_BORDER_W = 270
IMAGE_SAVE_PATH = ''

pathbig = '/home/shuzhilian/Jason/mmdetection/data/coco/val2017/'
dstpath = '/home/shuzhilian/Jason/Data/BOE_B7/testFile/result/'


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
    for i in range(0, len(labels)):
        if (cls_names[labels[i]] == 'NG' and bboxes[i][-1] > thd_ng) or cls_names[labels[i]] == 'OK':
            selectedCls.append(cls_names[labels[i]])
            selectedScore.append(bboxes[i][-1])
            tempBox = []
            tempBox = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
            selectedBox.append(tempBox)
    return selectedCls, selectedScore, selectedBox

def splitimage(src, rownum, colnum, dstpath):
    print(src)
    #img = cv2.imread((np.fromfile(src, dtype=np.uint8),cv2.IMREAD_GRAYSCALE))
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), -1)
    # print(img.shape)
    img = cv2.rectangle(img, (2520, 7808), (2840, 7908), [255], thickness=-1)
    constant = cv2.copyMakeBorder(img,0,ADD_BORDER_H,0,ADD_BORDER_W, cv2.BORDER_CONSTANT, value=255)
    # print(constant.shape)
    #img = Image.fromarray(cv2.cvtColor(constant, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(constant)
    w, h = img.size

    needHeight = ROW_HEIGHT + (ROW_HEIGHT - 20) * rownum
    needWidth = COL_WIDTH + (COL_WIDTH - 20) * colnum
    #if rownum <= h and colnum <= w:
    if needHeight <= h and needWidth <= w:
        # print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        # print('begin...')
        s = os.path.split(src)
        fn = s[1].split('.')
        # basename = fn[0]
        ext = fn[-1]
        num = 0
        for r in range(rownum + 2):
            for c in range(colnum + 2):
                if c == 0:
                    xStart = 0
                    xEnd = COL_WIDTH
                else:
                    xStart = c*COL_WIDTH-c*20
                    xEnd = xStart + COL_WIDTH
                if r == 0:
                    yStart = 0
                    yEnd = ROW_HEIGHT
                else:
                    yStart = r*ROW_HEIGHT-r*20
                    yEnd = yStart + ROW_HEIGHT
                box = (xStart, yStart, xEnd, yEnd)
                img.crop(box).save(dstpath + '/' + str(num + 1) + '.' + ext, ext)
                num = num + 1
        # print('finish %s pics。' % num)
    else:
        print('wrong')

def image_compose(image_column, image_row, pathSuffix):
    to_image = Image.new('RGB', (image_column * COL_WIDTH, image_row * ROW_HEIGHT))
    for y in range(0, image_row):
        for x in range(0, image_column):
            # print(x * COL_WIDTH, y * ROW_HEIGHT)
            # print(image_names[image_column * y + x])
            from_image = Image.open(pathSuffix + image_names[image_column * y + x])
            to_image.paste(from_image, (x * COL_WIDTH, y * ROW_HEIGHT))
    image_cut = to_image.crop((0, 0, to_image.size[0] - ADD_BORDER_W, to_image.size[1] - ADD_BORDER_H))
    return image_cut.save(IMAGE_SAVE_PATH)

def takeNum(elem):
    file = elem.split('/')[-1]
    return int(file.split('.')[0])

def takeNum_(elem):
    temp = elem.split('_')[-1]
    return int(temp.split('.')[0])

if __name__ == '__main__':
    # init model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    imgs = [pathbig + i for i in os.listdir(pathbig)]
    # logging for saving test result
    myPath = dstpath + 'totalStati'
    if os.path.exists(myPath) == False:
        os.makedirs(myPath)
    myPath_ = dstpath + 'temp'
    if os.path.exists(myPath_) == False:
        os.makedirs(myPath_)
    f = open(myPath + '/test.txt', 'a')
    labels = {}
    progress_bar = 0
    total_bar = len(imgs)
    for image in imgs:
        print('Current image index: ', progress_bar, '/', total_bar)
        progress_bar += 1
        # # fill image to mask out note pad region
        # splitimage(image, IMAGE_ROW, IMAGE_COLUMN, myPath_)
        # myPath_ = myPath_ + '/'
        # s_images = [myPath_ + i for i in os.listdir(myPath_)]
        # s_images.sort(key=takeNum)

        num = 0
        classes = ('OK', 'NG')
        result = inference_detector(model, image)
        selectedCls = []
        selectedScore = []
        selectedBox = []
        selectedCls, selectedScore, selectedBox = selectClsScoreBoxFromResult(result, classes, thd_ng=0.1)
        for index in range(0, len(selectedCls)):
            # print(selectedCls[index], 'score: ', selectedScore[index])
            if selectedCls[index] == 'NG':
                num += 1
        if num > 0:
            labels[image] = 'NG'
        else:
            labels[image] = 'OK'
        #####################################
    for image in imgs:
        f.writelines('\n' + image + ': ' + labels[image])
    print(labels)
