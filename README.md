# Weed Detection for Cotton Production Using Transformers

This repo contains the code for "Weed Detection for Cotton Production Using Transformers" my Msc dissertation submitted to the Department of Civil, Environmental and Geomatic Engineering at UCL.

## Dataset
### Data Preparation
- The data used in the study is from the CottonWeedDet12 Dataset available on [Zenodo](https://doi.org/10.5281/zenodo.7535814)
-  `commons/vig2yolov5.py` converts the data from VGG format to YOLO format
-  `commons/pationing_dataset_yolov5.py --outputDir` splits the data into training, validation and testing sets
-   `commons/yolov52coco.py` converts the data from YOLO format to COCO format

### Data Information
- `count_bbox_class.py` prints the number of bounding boxes for each class in the dataset
-  `detr-main/MEAN_STD.py` calculates the mean and standard deviation of the RGB channels for images in the dataset
	

## YOLOv4 Model Training and Testing

## Hyper Parameter Tuning for Transformers

## Transformers Training

## Transformers Testing

## Plotting logs

## Explainability

### Visualizing Self-Attention

### D-RISE Saliency Map Implementations


