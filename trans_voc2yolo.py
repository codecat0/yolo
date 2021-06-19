"""
@File  : trans_voc2yolo.py
@Author: CodeCat
@Time  : 2021/6/15 8:06
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil


voc_root = './VOCdevkit'
voc_version = 'VOC2012'

train_txt = 'train.txt'
val_txt = 'val.txt'

save_file_root = './my_yolo_data'

label_json = './pascal_voc_classes.json'

voc_images_path = os.path.join(voc_root, voc_version, 'JPEGImages')
voc_xml_path = os.path.join(voc_root, voc_version, 'Annotations')
train_txt_path = os.path.join(voc_root, voc_version, 'ImageSets', 'Main', train_txt)
val_txt_path = os.path.join(voc_root, voc_version, 'ImageSets', 'Main', val_txt)

assert os.path.exists(voc_images_path), 'VOC images path not exist'
assert os.path.exists(voc_xml_path), 'VOC xml path not exist'
assert os.path.exists(train_txt_path), 'VOC train txt path not exist'
assert os.path.exists(val_txt_path), 'VOC val txt path not exist'

if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)


def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names, save_root, class_dict, train_val='train'):
    sava_txt_path = os.path.join(save_root, train_val, 'labels')
    if os.path.exists(sava_txt_path) is False:
        os.makedirs(sava_txt_path)

    sava_images_path = os.path.join(save_root, train_val, 'images')
    if os.path.exists(sava_images_path) is False:
        os.makedirs(sava_images_path)

    for file in tqdm(file_names, desc='translate {} file...'.format(train_val)):
        img_path = os.path.join(voc_images_path, file + '.jpg')
        assert os.path.exists(img_path), 'file:{} not exist'.format(img_path)

        xml_path = os.path.join(voc_xml_path, file + '.xml')
        assert os.path.exists(xml_path), 'file:{} not exist'.format(img_path)

        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)['annotation']
        image_height = int(data['size']['height'])
        image_width = int(data['size']['width'])

        with open(os.path.join(sava_txt_path, file + '.txt'), 'w') as f:
            assert 'object' in data.keys(), "file: '{}' lack of object key".format(xml_path)
            for index, obj in enumerate(data['object']):
                xmin = float(obj['bndbox']['xmin'])
                ymin = float(obj['bndbox']['ymin'])
                xmax = float(obj['bndbox']['xmax'])
                ymax = float(obj['bndbox']['ymax'])
                class_name = obj['name']
                class_index = class_dict[class_name] - 1

                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some box w/h <=0 ".format(xml_path))
                    continue

                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                xcenter = round(xcenter / image_width, 6)
                ycenter = round(ycenter / image_height, 6)
                w = round(w / image_width, 6)
                h = round(h / image_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(' '.join(info))
                else:
                    f.write('\n' + ' '.join(info))

        shutil.copyfile(img_path, os.path.join(sava_images_path, img_path.split(os.sep)[-1]))


def create_dataset_yaml(class_names):
    train = os.path.join(save_file_root, 'train', 'images')
    val = os.path.join(save_file_root, 'val', 'images')
    nc = len(class_names)

    with open(os.path.join(save_file_root, 'data.yaml'), 'w') as f:
        f.write('train: ' + str(train) + '\n')
        f.write('val: ' + str(val) + '\n')
        f.write('nc: ' + str(nc) + '\n')
        f.write('names: ' + str(class_names))


def main():
    json_file = open(label_json, 'r')
    class_dict = json.load(json_file)

    with open(train_txt_path, 'r') as f:
        train_file_names = [i.strip('\n') for i in f.readlines() if len(i.strip()) > 0]

    translate_info(train_file_names, save_file_root, class_dict, 'train')

    with open(val_txt_path, 'r') as f:
        val_file_names = [i.strip('\n') for i in f.readlines() if len(i.split()) > 0]

    translate_info(val_file_names, save_file_root, class_dict, 'val')

    class_names = list(class_dict.keys())
    create_dataset_yaml(class_names)


if __name__ == '__main__':
    main()