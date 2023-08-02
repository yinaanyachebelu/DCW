# under CottonWeed_Detection
# run: python commons/vig2yolov5.py

import cv2
import os
from os import walk
import json
import sys
from shutil import copyfile

sys.path.append("../")

image_roots = ["datasets/0809_0820_2021_MEFAS", "datasets/06282021_MEFAS_NorthFarm",
                "datasets/07122021_MEFAS_NorthFarm",
                "datasets/08272021_MEFAS_BROOKSVILLE", "datasets/09032021_MEFAS_Brooksville",
               "datasets/09032021_MEFAS_WBAndrews", "datasets/09042021_MEFAS_WBAndrews",
               "datasets/09062021_MEFAS_WBAndrews", "datasets/09082021_MEFAS_WBAndrews",
               "datasets/09092021_MEFAS_WBAndrews", "datasets/09102021_MEFAS_WBAndrews",
               "datasets/09122021_MEFAS_WBAndrews", "datasets/09132021_MEFAS_NorthFarm"]
# image_roots = ["datasets/09042021_MEFAS_WBAndrews"]

out_dir = "datasets/CottonWeedDataYolov5"
json_path = 'commons/class_indices_top12.json'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + "/images", exist_ok=True)
os.makedirs(out_dir + "/labels", exist_ok=True)
os.makedirs(out_dir + "/labels_json", exist_ok=True)

files = []
ext = (".jpeg", ".jpg", ".png", ".PNG")

for image_root in image_roots:
    print("Image Folders: ", image_root)
    for (dirpath, dirnames, filenames) in walk(image_root):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(dirpath, filename))

length = 0
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
json_file = open(json_path, "r")
class_dict = json.load(json_file)

print("Working...")
for file in files:
    length += 1
    if length % 1000 == 0:
        print("Working on the {}th image...".format(length))

    img = cv2.imread(file)
    cv2.imwrite(os.path.join(out_dir, "images", os.path.basename(file)), img)

    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]

    # read the corresponding json files
    json_file1 = open(file.split('.')[0] + '.json', "r")
    label_indict = json.load(json_file1)

    copyfile(file.split('.')[0] + '.json',
             out_dir + "/labels_json/" + file.split('/')[-1][:-4] + '.json')


    if isinstance(list(label_indict.values())[0]['regions'], dict):
        region = list(label_indict.values())[0]['regions']
        x_min = region['shape_attributes']['x']
        y_min = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        x = (x_min + w / 2) / width
        y = (y_min + h / 2) / height
        w = w / width
        h = h / height

        label = list(region['region_attributes']['CottonWeed'].keys())[0]
        if label not in class_dict:
            # remove the image since it has only an "Unknown label"
            print("The image contains only an Unknown label: ", file)
            os.remove(os.path.join(out_dir, "images", os.path.basename(file)))
            length -= 1
        else:
            weed_class = class_dict[label]
            txt_path = os.path.join(out_dir, "labels", file.split('/')[-1].split('.')[0] + '.txt')
            with open(txt_path, 'w', encoding='UTF-8') as f:
                f.write('{} {} {} {} {}'.format(weed_class, x, y, w, h))
                f.write('\n')
    else:
        regions = list(label_indict.values())[0]['regions']
        len_labels = len(regions)
        len_label = 0

        for idx, r in enumerate(regions):
            label = list(r['region_attributes']['CottonWeed'].keys())[0]
            if label not in class_dict:
                print("The image contains Unknown labels : ", file)
                os.remove(os.path.join(out_dir, "images", os.path.basename(file)))
                length -= 1
                break
        else:
            for r in regions:
                x_min = r['shape_attributes']['x']
                y_min = r['shape_attributes']['y']
                w = r['shape_attributes']['width']
                h = r['shape_attributes']['height']

                x = (x_min + w / 2) / width
                y = (y_min + h / 2) / height
                w = w / width
                h = h / height

                label = list(r['region_attributes']['CottonWeed'].keys())[0]
                weed_class = class_dict[label]
                txt_path = os.path.join(out_dir, "labels", file.split('/')[-1].split('.')[0] + '.txt')

                with open(txt_path, 'a', encoding='UTF-8') as f:
                    f.write('{} {} {} {} {}'.format(weed_class, x, y, w, h))
                    f.write('\n')

print("----------Summary----------")
print("Overall images: ", length)
print("----------Summary----------")
