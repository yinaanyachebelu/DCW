from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoeval import Params
#from IPython.display import JSON
import numpy as np
from Evaluator import evaluate_metrics, display_metrics
import argparse


def _compute_and_display_metrics(args):
    coco_gt = COCO(args.gt_coco_path,)
    coco_pred = coco_gt.loadRes(args.evaluation_result_path)
    cocoEval = COCOeval(coco_gt, coco_pred, "bbox")
    # Load the default parameters for COCOEvaluation
    params = cocoEval.params

    # Modify required parameters. Available params are:
    #imgIds          - [all],
    #catIds          - [all],
    # iouThrs         - [.5:.05:.95],
    # areaRng,maxDets - [1 10 100],
    #iouType         - ['bbox'],useCats
    # eg. param.iouType = 'bbox'
    #params.iouThrs = np.linspace(.5, .9, int(np.round((.9 - .5) / .1)) + 1, endpoint=True)

    # Calculate the metrics
    precision, recall, scores, iou_lookup = evaluate_metrics(
        cocoEval, params, args.show_eval_summary)

    # take precision for all classes, all areas and 100 detections
    # display_metrics(precision, recall, scores, iou_lookup,
    #                 log_path=args.output_log_path)

    # Calculate metrics for each category
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        # Calculate the metrics
        params.catIds = [cat["id"]]
        precision, recall, scores, iou_lookup = evaluate_metrics(
            cocoEval, params, args.show_eval_summary)
        # take precision for all classes, all areas and 100 detections
        # display_metrics(precision, recall, scores, iou_lookup,
        #                 class_name=cat["name"], log_path=args.output_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Metrics from the predictions and Ground Truths")
    parser.add_argument("--gt_coco_path", type=str, required=True)
    parser.add_argument("--evaluation_result_path", type=str, required=True)
    # parser.add_argument("--output_log_path", type=str,
    #                     default="evaluation.log")
    parser.add_argument("--show_eval_summary", type=bool, default=False)

    args = parser.parse_args()
    _compute_and_display_metrics(args)
