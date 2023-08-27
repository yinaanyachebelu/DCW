import os
import torchvision

dataset = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/'

#ANNOTATION_FILE_NAME = "annotations.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train2017")
VAL_DIRECTORY = os.path.join(dataset, "val2017")
TEST_DIRECTORY = os.path.join(dataset, "test2017")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        ann_path: str,
        image_processor,
        train: bool = True

    ):
        annotation_file_path = os.path.join(image_directory_path, ann_path)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


if __name__ == "__main__":

    TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor,
                                  ann_path='/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_train2017.json', train=True)
    VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor,
                                ann_path='/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_val2017.json', train=False)
    TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor,
                                 ann_path='/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_0_COCO_format/annotations/instances_test2017.json', train=False)

    print("Number of training examples:", len(TRAIN_DATASET))
    print("Number of validation examples:", len(VAL_DATASET))
    print("Number of test examples:", len(TEST_DATASET))
