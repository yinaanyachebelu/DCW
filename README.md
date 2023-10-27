# Weed Detection for Cotton Production Using Transformers

This repo contains the code for "Weed Detection for Cotton Production Using Transformers", my Msc dissertation submitted to the Department of Civil, Environmental and Geomatic Engineering at UCL.

## Dataset
### Data Preparation
- The data used in the study is from the CottonWeedDet12 Dataset available at [Zenodo](https://doi.org/10.5281/zenodo.7535814)
-  `commons/vig2yolov5.py` converts the data from VGG format to YOLO format
-  `commons/pationing_dataset_yolov5.py` splits the data into training, validation and testing sets
-   `commons/yolov52coco.py` converts the data from YOLO format to COCO format

### Data Information
- `count_bbox_class.py` prints the number of bounding boxes for each class in the dataset
-  `detr-main/MEAN_STD.py` calculates the mean and standard deviation for each channel for images in the dataset
	

## Baseline YOLOv4 Model Training and Testing
- `train_cuda0.sh` trains the YOLOv4 format with parameters set by [Dang et. al. 2023](https://doi.org/10.1016/j.compag.2023.107655)
-  Use the `coco2yolo` library to turn the COCO format files to YOLO format
	- This makes a copy of the dataset in YOLO format where the image files are named based on their order in the dataset for for testing
- `test_coco.py` tests the YOLOv4 model on the test set and saves the outputs as a json file

## Hyperparameter Tuning for Transformers
- `detr-main/param.py` conducts hyperparameter tuning using the Optuna library for the DETR model
- `deformable-detr/tuning.py` conducts hyperparameter tuning using the Optuna library for the Deformable-DETR model

## Transformers Training
The scripts for the DETR and Deformable DETR training are edited versions of the code from Github repositories of their original implementations by Facebook AI and Fundamental Vision.
- To train the DETR model, run `python detr-main/main.py --dataset_file weed --data_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ --output_dir  /home/ayina/MscThesis/DCW/detr-main/runs --resume weights/detr-r50-e632da11.pth` 
- To train the Deformable DETR model, run `python -u main_scratch.py  --with_box_refine --two_stage  --coco_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ `
- To train Deformable DETR model with CIoU replacing GIoU loss: 
	- Code for CIoU Implementation is derived from Torchvision [source code](https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html)
	- In deformable-detr/models/__init__.py, change first line of code from *.deformable_detr import build* to *.deformable_detr_FOCAL import build*. This will use a different set of scripts with the CIoU implementation code to build the model.
	- Run training normally (see previous bullet point).

## Transformers Testing
- To test the DETR model, run `python detr-main/main_test.py --batch_size 2 --no_aux_loss --eval --resume [PATH TO CHECKPOINT] --dataset_file weed --data_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/`
- To test the Deformable DETR model, run `python -u deformable-detr/main_test.py --batch_size 4 --no_aux_loss --eval --resume [PATH TO CHECKPOINT] --with_box_refine --two_stage --coco_path /home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/ --save_dir 'preds/predictions_focal_laprop4_87_withlaprop.json' `
- To evaluate the performance of the models (mAP and mAR), run `inference.py`
- To evaluate the performance of the models *by class*, run `inference_classes.py`

## Visualizations
- To visualize the output of the transformer models on sample test images, run `python detr-main/test_viz.py` or `python deformable-detr/test_viz.py`
- To visualize the output of all 3 models on the test images and analyze errors, use notebook `fiftyone.ipynb` 
### Plotting Logs
- `detr-main/plotting.py` plots the training loss curves for the DETR model
-  The notebook `detr-main/plot_train_detr.ipynb` also plots training loss curves for the DETR model
- `deformable-detr/plotting.py` plots the training loss curves for the Deformable DETR model
- The notebook `deformable-detr/plot_train_deformable.ipynb` also plots training loss curves for the Deformable DETR model

## Explainability

### Visualizing Self-Attention
- `detr-main/attention.py` visualizes the self-attention module for the DETR model 

### D-RISE Saliency Map Implementations
The code to create the D-RISE saliency maps to explain the outputs of the YOLOv4 and deformable DETR model are partially derived from the this [Github repository](https://github.com/RuoyuChen10/objectdetection-saliency-maps/blob/main/tutorial/drise-yolov3.ipynb)
- The implementation of D-RISE explanations for the Yolov4 model can be found in the `YOLOv4/drise_yolo.ipynb` notebook
- The implementation of D-RISE explanations for the Deformable DETR model can be found in the `deformable-detr/drise_detr.ipynb` notebook

## References
- Benchmarking of CottonWeedDet12 dataset including YOLOv4: https://github.com/DongChen06/DCW
- DETR Implementation by Facebook AI: https://github.com/facebookresearch/detr
- Deformable DETR Implementation by Fundamental Vision: https://github.com/fundamentalvision/Deformable-DETR
- D-RISE Original Paper: https://arxiv.org/pdf/2006.03204.pdf


