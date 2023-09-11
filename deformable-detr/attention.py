# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_test import evaluate, train_one_epoch
from models import build_model
import matplotlib.pyplot as plt
import torchvision.transforms as T
from datasets.coco import make_coco_transforms


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


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1.9e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1.9e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

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
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4,
                        type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False,
                        action='store_true', help='whether to cache images on memory')
    # By default, Model was trained on 91 classes
    parser.add_argument('--num_classes', default=13, type=int)

    return parser


@torch.no_grad()
def evaluate_test(model, criterion, postprocessors, data_loader, device, thres=0.8):
    model.eval()
    criterion.eval()
    thresh = 0.9

    img_61 = "/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/test2017/000000000061.jpg"
    img_22 = "/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/test2017/000000000022.jpg"

    orig_image = Image.open(img_22)
    w, h = orig_image.size
    transform = make_coco_transforms("test")
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    image, targets = transform(orig_image, dummy_target)
    image = image.unsqueeze(0)
    image = image.to(device)

    # using hooks to extract attention weights
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)

        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output)

        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output)

        ),

    ]

    outputs = model(image)

    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > thresh

    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep], orig_image.size)

    for hook in hooks:
        hook.remove()

    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0].cpu()
    dec_attn_weights = dec_attn_weights[0].cpu()

    f_map = conv_features['0']
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map.tensors.shape)
    print(" ")
    print("Decoder attention:      ", dec_attn_weights[0].shape)

    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]

    shape_22a = [251, 71]
    shape_22b = [16, 16]

    shape_22c = [71, 251]
    shape_22d = [32, 8]

    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape_22c + shape_22d)
    print("Reshaped self-attention:", sattn.shape)

    fact = 32

    # let's select 4 reference points for visualization
    idxs = [(200, 200), (280, 400)]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    # and we add one plot per reference point
    gs = fig.add_gridspec(1, 3)
    axs = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2])
    ]

    # for each one of the reference points, let's plot the self-attention

    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]],
                  cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[0, 0])
    fcenter_ax.imshow(orig_image)
    for (y, x) in idxs:
        scale = orig_image.height / image.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle(
            (x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')
    fig.savefig('charts/blue_att_3.jpg')

#######################################################

    # visualizing attention

    # # taken from FB Research DETR hands-on tutorial notebook: https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=hYVZjfGhYTEa

    # h, w = conv_features['0'].tensors.shape[-2:]

    # fig, axs = plt.subplots(ncols=len(bboxes_scaled),
    #                         nrows=2, figsize=(22, 9.5))
    # colors = COLORS * 100
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     img = np.array(orig_image)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     ax.imshow(img)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(cats[probas[idx].argmax()])
    # fig.tight_layout()
    # plt.savefig('charts/22_att_deform.jpg')


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
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
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(
                dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(
                dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_test)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # LOAD WEIGHTS INTO MODEL
            checkpoint = torch.load(args.resume, map_location='cpu')
            # When number of classes changes, modify the model as well. Otherwise, keep original weights !
            if args.num_classes != 13 or args.dataset_file != 'coco':
                print(
                    f"Deleting last linear layer weights as num_classes is different {args.num_classes} than expected for coco (91)")
                keys = list(checkpoint['model'].keys())
                for i in keys:
                    if 'class_embed' in i:
                        del checkpoint["model"][i]
            else:
                print("Keeping all the original weights.")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (
            k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # import pdb; pdb.set_trace()
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            # p_groups = copy.deepcopy(optimizer.param_groups)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # for pg, pg_old in zip(optimizer.param_groups, p_groups):
            #     pg['lr'] = pg_old['lr']
            #     pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator, val_loss = evaluate(
                model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir
            )
    args.eval = True
    if args.eval:

        evaluate_test(model, criterion, postprocessors,
                      data_loader_test, device, thres=0.8)

        print("IMAGE SAVED!!")

        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
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

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                  **{f'test_{k}': v for k, v in test_stats.items()},
    #                  'epoch': epoch,
    #                  'n_parameters': n_parameters}

    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #         # for evaluation logs
    #         if coco_evaluator is not None:
    #             (output_dir / 'eval').mkdir(exist_ok=True)
    #             if "bbox" in coco_evaluator.coco_eval:
    #                 filenames = ['latest.pth']
    #                 if epoch % 50 == 0:
    #                     filenames.append(f'{epoch:03}.pth')
    #                 for name in filenames:
    #                     torch.save(coco_evaluator.coco_eval["bbox"].eval,
    #                                output_dir / "eval" / name)

    # total_time = time.time() - start_time
    # total_time_str = str(time.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
