from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import argparse

ann_path = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_test2017.json'
pred_path_yolo = '/home/ayina/MscThesis/DCW/YOLOv4/runs/test/exp9/best_predictions.json'
pred_path_detr = '/home/ayina/MscThesis/DCW/detr-main/preds/test_predictions.json'
pred_path_deform = '/home/ayina/MscThesis/DCW/deformable-detr/preds/test_predictions.json'

# testing using val set with detr
ann_path_val = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_val2017.json'
pred_path_detr_val = '/home/ayina/MscThesis/DCW/detr-main/preds/model_predictions.json'


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Get Coco json results by class', add_help=False)
    parser.add_argument(
        '--pred_path', default='/home/ayina/MscThesis/DCW/deformable-detr/preds/test_predictions.json', type=str)

    return parser


columns = ['Category', 'AP50', 'AP@[0.50:0.95]', 'AR@[0.50:0.95]']
results = []

ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cats = [
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


def main():

    pred_path = args.pred_path

    for id in ids:
        num = id - 1
        cat = cats[num]

        Gt = COCO(ann_path)
        dets = Gt.loadRes(pred_path)

        coco_eval = COCOeval(Gt, dets, "bbox")
        coco_eval.params.catIds = id
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_50 = round(coco_eval.stats[1], 3)
        ap_5095 = round(coco_eval.stats[0], 3)
        aR = round(coco_eval.stats[7], 3)

        results.append([cat, ap_50, ap_5095, aR])

    results_df = pd.DataFrame(results, columns=columns)
    # results_df.to_csv('results/detr_classes.csv')
    print(results_df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Get Coco json results by class', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
