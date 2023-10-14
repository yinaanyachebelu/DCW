import json

# Specify the path to your JSON file
ann_path= "/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_train2017.json"

# Read the JSON file and load its contents into a Python dictionary
with open(ann_path, "r") as json_file:
    ann= json.load(json_file)
    
data = ann["annotations"]
    
class_counts = {}

# Iterate through the data and count bounding boxes for each class
for item in data:
    category_id = item["category_id"]
    if category_id in class_counts:
        class_counts[category_id] += 1
    else:
        class_counts[category_id] = 1

# Print or save the class counts
for category_id, count in class_counts.items():
    print(f"Class {category_id}: {count} bounding boxes")