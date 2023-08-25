from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann_path = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_test2017.json'
pred_path = '/home/ayina/MscThesis/DCW/YOLOv4/runs/test/exp9/best_predictions.json'

ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def main():
    for cat in ids:
        Gt = COCO(ann_path)
        dets = Gt.loadRes(pred_path)

        coco_eval = COCOeval(Gt, dets, "bbox")
        coco_eval.params.catIds = cat
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    main()
