
##########

# code to find mean and std for dataset derived from : https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.google_utils import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *
import warnings
warnings.filterwarnings('ignore')


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


cats = [
    "N/A",
    "Waterhemp",
    "MorningGlory",
    "Purslane",
    "SpottedSpurge",
    "Carpetweed",
    "Ragweed",
    "Eclipta",
    "PricklySida",
    "PalmerAmaranth",
    "Sicklepod",
    "Goosegrass",
    "CutleafGroundcherry"
]


def mean_std(data,
             weights=None,
             batch_size=6,
             imgsz=640,
             conf_thres=0.001,
             iou_thres=0.6,  # for NMS
             save_json=True,
             single_cls=False,
             augment=False,
             verbose=False,
             model=None,
             dataloader=None,
             save_dir=Path(''),  # for saving images
             save_txt=False,  # for auto-labelling
             save_conf=False,
             plots=True,
             log_imgs=0,
             max_size=640):  # number of logged images

    # Initialize/load model and set device
    #training = model is not None

    device = select_device(opt.device, batch_size=batch_size)

    imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Configure
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # Dataloader

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    path = data['train']
    dataloader = create_dataloader(
        path, imgsz, batch_size, 64, opt, pad=0, rect=True)[0]

    seen = 0
    ######################################################################

    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        print(img.shape)


# def batch_mean_and_sd(loader):

#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)

#     for images, _ in loader:
#         print(images.shape)
#     #     b, c, h, w = images.shape
#     #     nb_pixels = b * h * w
#     #     sum_ = torch.sum(images, dim=[0, 2, 3])
#     #     sum_of_square = torch.sum(images ** 2,
#     #                               dim=[0, 2, 3])
#     #     fst_moment = (cnt * fst_moment + sum_) / (
#     #                   cnt + nb_pixels)
#     #     snd_moment = (cnt * snd_moment + sum_of_square) / (
#     #                         cnt + nb_pixels)
#     #     cnt += nb_pixels

#     # mean, std = fst_moment, torch.sqrt(
#     #   snd_moment - fst_moment ** 2)
#     #    return mean, std

if __name__ == '__main__':

    # mean, std = batch_mean_and_sd(data_loader_train)
    # print("mean and std: \n", mean, std)
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/ayina/MscThesis/DCW/YOLOv4/runs/train/yolov4_08/weights/best.pt')
    parser.add_argument(
        '--data', type=str, default='./data/cottonweedsdetection_seed0.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='600', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true',
                        help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--verbose', action='store_true',
                        help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true',
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test',
                        help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str,
                        default='./cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str,
                        default='./data/weed.names', help='*.cfg path')
    opt = parser.parse_args()

    mean_std(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )
