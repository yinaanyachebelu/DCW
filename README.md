# Weed Detection for Cotton Production Using Transformers

This repo contains the code for "Weed Detection for Cotton Production Using Transformers" my Msc dissertation submitted to the Department of Civil, Environmental and Geomatic Engineering at UCL.

## Dataset
### Data Preparation
- The data used in the study is from the CottonWeedDet12 Dataset available at [Zenodo](https://doi.org/10.5281/zenodo.7535814)
-  `commons/vig2yolov5.py` converts the data from VGG format to YOLO format
-  `commons/pationing_dataset_yolov5.py` splits the data into training, validation and testing sets
-   `commons/yolov52coco.py` converts the data from YOLO format to COCO format

### Data Information
- `count_bbox_class.py` prints the number of bounding boxes for each class in the dataset
-  `detr-main/MEAN_STD.py` calculates the mean and standard deviation for each channel for images in the dataset
	

## Baseline YOLO Model Training and Testing
- `train_cuda0.sh` trains the YOLOv4 format with parameters set by [Dang et. al. 2023](https://doi.org/10.1016/j.compag.2023.107655)
-  Use the `coco2yolo` library to turn the turn the COCO format files to YOLO format
	- This makes a copy of the dataset in YOLO format where the image files are named based on their order in the dataset for for testing
- `test_coco.py` tests the YOLOv4 model on the test set and saves the outputs as a json file

## Hyper Parameter Tuning for Transformers
- `detr-main/param.py` conducts hyperparameter tuning using the Optuna library for the DETR model
- `deformable-detr/tuning.py` conducts hyperparameter tuning using the Optuna library for the Deformable-DETR model

## Transformers Training
The scripts for the DETR and Deformable DETR training are edited versions of the code from Github repositories of their original implementations by Facebook AI and Fundamental Vision.
- To train the DETR model, run `python detr-main/main.py --dataset_file weed --data_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ --output_dir  /home/ayina/MscThesis/DCW/detr-main/runs --resume weights/detr-r50-e632da11.pth` directory
- To train the Deformable DETR model, run `python -u main.py --output_dir runs/ --with_box_refine --two_stage --resume ./weights/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth --coco_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ --num_classes=13`

## Transformers Testing
- To test the DETR model, run `python detr-main/main_test.py --batch_size 2 --no_aux_loss --eval --resume runs/checkpoint.pth --dataset_file weed --data_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/`
- To test the Deformable DETR model, `python -u deformable-detr/main_test.py --batch_size 4 --no_aux_loss --eval --resume ./runs_focal_laprop_4/checkpoint0069.pth --with_box_refine --two_stage --coco_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ --save_dir 'preds/predictions_focal_laprop4_87_withlaprop.json' --laprop`
- To evaluate the performance of the models (mAP and mAR) based on their outputs on the test set, run `inference.py`
- To evaluate the performance of the models *by class*, run `inference_classes.py`

## Plotting logs

## Explainability

### Visualizing Self-Attention

### D-RISE Saliency Map Implementations

## References


