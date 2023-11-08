# Training script on CUDA 0
# bash -i train_cuda0.sh

# YOLOv4
# cd /home/ayina/MscThesis/DCW/YOLOv4;  /home/ayina/anaconda3/envs/cottonweeddetection/bin/python   test.py --task 'test' --device 0 --batch-size 8 --img 640 --data cottonweedsdetection_test.yaml --cfg cfg/yolov4.cfg --weights runs/train/yolov4_08/weights/best.pt


cd /home/ayina/MscThesis/DCW/YOLOv4;  /home/ayina/anaconda3/envs/cottonweeddetection/bin/python   test_coco.py --task 'test' --device 0 --batch-size 8 --img 640 --data test_coco.yaml --cfg cfg/yolov4.cfg --weights runs/train/yolov4_010/weights/best.pt
