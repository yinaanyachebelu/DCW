from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoeval import Params
import numpy as np
import logging


def evaluate_metrics(cocoEval, params=None, display_summary=False):
    # Display the iouThresholds for which the evaluation took place
    if params:
        cocoEval.params = params
    print("IoU Thresholds: ", cocoEval.params.iouThrs)
    iou_lookup = {float(format(val, '.2f')): index for index,
                  val in enumerate(cocoEval.params.iouThrs)}

    cocoEval.evaluate()  # Calculates the metrics for each class
    # Stores the values in the cocoEval's 'eval' object
    cocoEval.accumulate(p=params)
    if display_summary:
        cocoEval.summarize()  # Display the metrics.

    # Extract the metrics from accumulated results.
    precision = cocoEval.eval["precision"]
    recall = cocoEval.eval["recall"]
    scores = cocoEval.eval["scores"]

    return precision, recall, scores, iou_lookup


# Print final results
def display_metrics(precision, recall, scores, iou_lookup, class_name=None, log_path='evaluation.txt'):
    # Initialize logger
    logger = logging.getLogger('eval_log')
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

    if class_name:
        logger.warning(
            "| Class Name | IoU | mAP | F1-Score | Precision | Recall |")
        logger.warning(
            "|------------|-----|-----|----------|-----------|--------|")
    else:
        logger.warning("| IoU | mAP | F1-Score | Precision | Recall |")
        logger.warning("|-----|-----|----------|-----------|--------|")

    for iou in iou_lookup.keys():
        precesion_iou = precision[iou_lookup[iou], :, :, 0, -1].mean(1)
        scores_iou = scores[iou_lookup[iou], :, :, 0, -1].mean(1)
        recall_iou = recall[iou_lookup[iou], :, 0, -1]
        prec = precesion_iou.mean()
        rec = recall_iou.mean()

        if class_name:
            # print("Class Name: {:10s} IoU: {:2.2f} mAP: {:6.3f} F1-Score: {:2.3f} Precision: {:2.2f} Recall: {:2.2f}".format(
            #    class_name, iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec
            # ))
            logger.warning("|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                class_name, iou, prec * 100, scores_iou.mean(), (2 * prec * rec /
                                                                 (prec + rec + 1e-8)), prec, rec
            ))
        else:
            # print("IoU: {:2.2f} mAP: {:6.3f} F1-Score: {:2.3f} Precision: {:2.2f} Recall: {:2.2f}".format(
            #    iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec
            # ))

            logger.warning("|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                iou, prec * 100, scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec))
