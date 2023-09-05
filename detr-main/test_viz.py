# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py 
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils

from models import build_model
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.weed_coco import make_Weed_transforms
#import datasets.transforms as T
import torchvision.transforms as T
from engine_viz import evaluate, train_one_epoch

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, COLORS, CLASSES):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    plt.savefig("graphics/test_graphic.jpg")


def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-05, type=float)
    parser.add_argument('--lr_backbone', default=5e-06, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=5e-05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.2, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='runs2/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def prepare_viz(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


@torch.no_grad()
def evaluate_test(model, criterion, postprocessors, data_loader, device, thres=0.8):
    model.eval()
    criterion.eval()
    thresh = 0.9

    # colors for vizualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    cats = [
        "N/A"
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

    n = 6
    imgs = get_images("/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/test2017/")
    imgs_selected = random.sample(imgs, n)
    print(imgs_selected)

    fig = plt.figure(figsize=(30, 10))

    for i, img in enumerate(imgs_selected):

        #url = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/test2017/000000000183.jpg'

        orig_image = Image.open(img)
        w, h = orig_image.size
        transform = make_Weed_transforms("test")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        outputs = model(image)

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > thresh

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()

        # plotting

        fig.add_subplot(2, 3, i + 1)
        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.gca()
        ax.imshow(img)

        for score, box in zip(probas, bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)

            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, color='r', linewidth=3))
            cl = score.argmax()
            text = f'{cats[cl]}: {score[cl]:0.2f}'
            ax.text(bbox[0], bbox[1] - 35, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
    plt.savefig('graphics/plt_test_graphic_multi.jpg',)

    # for idx, box in enumerate(bboxes_scaled):
    #     bbox = box.cpu().data.numpy()
    #     bbox = bbox.astype(np.int32)
    #     bbox = np.array([
    #         [bbox[0], bbox[1]],
    #         [bbox[2], bbox[1]],
    #         [bbox[2], bbox[3]],
    #         [bbox[0], bbox[3]],
    #     ])
    #     bbox = bbox.reshape((4, 2))
    #     cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

    #     cv2.imwrite('graphics/test_graphic1.jpg', img)

    #imgs = get_images("/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/test2017/")

    #fig = plt.subplots(2, 4, figsize=(26, 17))
    #fig = plt.figure(figsize=(26, 17))

    # for samples, targets in data_loader:

    #     samples = samples.to(device)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     outputs = model(samples)

    #     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #     results, results_nodict = postprocessors['bbox'](outputs, orig_target_sizes)

    #     # if 'segm' in postprocessors.keys():
    #     #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #     #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

    #     res = {target['image_id'].item(): output for target, output in zip(targets, results)}
    #     print("new image")
    # print(res)

    #plt.imshow(samples.permute(1, 2, 0))
    #ax = fig.add_subplot(2, 4, i + 1)
    # ax.imshow(samples)
    # for original_id, prediction in res.items():
    #     if len(prediction) == 0:
    #         continue

    #     boxes = prediction["boxes"].tolist()
    #     scores = prediction["scores"].tolist()
    #     labels = prediction["labels"].tolist()

    #     bsl = zip(boxes, scores, labels)

    #     color = (0, 0, 220)
    #     image = cv2.rectangle(samples,
    #                              (box[0], box[1]),
    #                               (box[2] + b[0], b[3] + b[1]),
    #                            color, 1)

    #     print("new image:")
    #     print("boxes")
    #     print(boxes)
    #     print("socres")
    #     print(scores)
    #     print("labels")
    #     print(labels)

    #     fig.add_subplot(2, 4, i + 1)
    #     for s, l, b in results_nodict:

    #         if s > thres:
    #             box = b.tolist()
    #             color = (0, 0, 220)  # if p>0.5 else (0,0,0)
    #             image = cv2.rectangle(samples,
    #                                   (box[0], box[1]),
    #                                   (box[2] + b[0], b[3] + b[1]),
    #                                   color, 1)
    #             plt.imshow(image)
    #             plt.axis('off')

    # fig.tight_layout()
    # plt.savefig("graphics/test_viz.jpg")

    return


# @torch.no_grad()
# def infer(images_path, model, postprocessors, device, output_path):
#     model.eval()
#     duration = 0
#     for img_sample in images_path:
#         filename = os.path.basename(img_sample)
#         print("processing...{}".format(filename))
#         orig_image = Image.open(img_sample)
#         w, h = orig_image.size
#         transform = make_Weed_transforms("val")
#         dummy_target = {
#             "size": torch.as_tensor([int(h), int(w)]),
#             "orig_size": torch.as_tensor([int(h), int(w)])
#         }
#         image, targets = transform(orig_image, dummy_target)
#         image = image.unsqueeze(0)
#         image = image.to(device)

#         conv_features, enc_attn_weights, dec_attn_weights = [], [], []
#         hooks = [
#             model.backbone[-2].register_forward_hook(
#                 lambda self, input, output: conv_features.append(output)

#             ),
#             model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
#                 lambda self, input, output: enc_attn_weights.append(output[1])

#             ),
#             model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
#                 lambda self, input, output: dec_attn_weights.append(output[1])

#             ),

#         ]

#         start_t = time.perf_counter()
#         outputs = model(image)
#         end_t = time.perf_counter()

#         outputs["pred_logits"] = outputs["pred_logits"].cpu()
#         outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

#         probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#         # keep = probas.max(-1).values > 0.85
#         keep = probas.max(-1).values > args.thresh

#         bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
#         probas = probas[keep].cpu().data.numpy()

#         for hook in hooks:
#             hook.remove()

#         conv_features = conv_features[0]
#         enc_attn_weights = enc_attn_weights[0]
#         dec_attn_weights = dec_attn_weights[0].cpu()

#         # get the feature map shape
#         h, w = conv_features['0'].tensors.shape[-2:]

#         if len(bboxes_scaled) == 0:
#             continue

#         img = np.array(orig_image)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         for idx, box in enumerate(bboxes_scaled):
#             bbox = box.cpu().data.numpy()
#             bbox = bbox.astype(np.int32)
#             bbox = np.array([
#                 [bbox[0], bbox[1]],
#                 [bbox[2], bbox[1]],
#                 [bbox[2], bbox[3]],
#                 [bbox[0], bbox[3]],
#             ])
#             bbox = bbox.reshape((4, 2))
#             cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

#         # img_save_path = os.path.join(output_path, filename)
#         # cv2.imwrite(img_save_path, img)
#         cv2.imshow("img", img)
#         cv2.waitKey()
#         infer_time = end_t - start_t
#         duration += infer_time
#         print("Processing...{} ({:.3f}s)".format(filename, infer_time))

#     avg_duration = duration / len(images_path)
#     print("Avg. Time: {:.3f}s".format(avg_duration))

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # NEW DATALOADER FOR TEST VIZUALIZATION
    indices = [4, 10, 75, 100, 460, 800]
    subset = torch.utils.data.Subset(dataset_test, indices)

    data_loader_test = DataLoader(subset, args.batch_size, sampler=torch.utils.data.SequentialSampler(subset),
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        #del checkpoint["model"]["class_embed.weight"]
        #del checkpoint["model"]["class_embed.bias"]
        #del checkpoint["model"]["query_embed.weight"]

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        #model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:

        evaluate_test(model, criterion, postprocessors, data_loader_test, device, thres=0.8)

        print("IMAGE SAVED!!")

        # test_stats, coco_evaluator, val_loss = evaluate(model, criterion, postprocessors,
        #                                                 data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator, val_loss = evaluate(
            model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    # model, criterion, postprocessors = build_model(args)
    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    # model.to(device)
    # image_paths = get_images(args.data_path)

    #infer(image_paths, model, postprocessors, device, args.output_dir)
