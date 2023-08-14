# import the necessary packages
from glob import glob
YOLO_CONFIG = "darknet/cfg/yolov4.cfg"
YOLO_WEIGHTS = "data/yolov4.weights"
COCO_DATA = "darknet/cfg/coco.data"
YOLO_NETWORK_WIDTH = 608
YOLO_NETWORK_HEIGHT = 608
LABEL2IDX = "data/label2idx.pkl"
YOLO_90CLASS_MAP = "data/yolo_90_class_map.pkl"
IMAGES_PATH = glob("data/val2017/*")
COCO_GT_ANNOTATION = "data/annotations/instances_val2017.json"
COCO_VAL_PRED = "data/COCO_Val_Predictions.json"
CONF_THRESHOLD = 0.25
IOU_GT = [90, 80, 250, 450]
IOU_PRED = [100, 100, 220, 400]
IOU_RESULT = "results/dog_iou.png"
PR_RESULT = "results/pr_curve.png"
GROUND_TRUTH_PR = ["dog", "cat", "cat", "dog", "cat", "cat", "dog",
  "dog", "cat", "dog", "dog", "dog", "dog", "cat", "cat", "dog"]
PREDICTION_PR = [0.7, 0.3, 0.5, 0.6, 0.3, 0.35, 0.55, 0.2, 0.4, 0.3,
  0.7, 0.5, 0.8, 0.2, 0.9, 0.4]