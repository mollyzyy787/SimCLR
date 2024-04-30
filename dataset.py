from torchvision.datasets import CocoDetection, VisionDataset
from collections import defaultdict
import torch
from simclr.modules.transformations import TransformsSimCLR
from pycocotools.coco import COCO
import cv2
from PIL import Image

import os

class CustomCocoClassification(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.images = []
        self.targets = []
        self.transform = transform

        # Iterate through the root directory
        for root_dir, _, filenames in os.walk(root):
            for filename in filenames:
                # Check if the file is a JPEG image
                if filename.lower().endswith('.jpg'):
                    # Parse the image filename to extract category ID and unique ID
                    _, uid, category_id = filename[:-4].split('_')
                    image_path = os.path.join(root_dir, filename)
                    self.images.append(image_path)
                    if category_id == '41':
                        self.targets.append(12)
                    else:
                        self.targets.append(int(category_id)-1)  # Convert category ID to integer
    
    def __getitem__(self, idx):
        try:
            image_path, target = self.images[idx], self.targets[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, target
        except FileNotFoundError:
            # If file is not found, print a message and return None
            print(f"File not found for index {idx}")
            return None
        
    def __len__(self):
        return len(self.images)
        
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except FileNotFoundError:
            # If file is not found, print a message and return None
            print(f"File not found for index {idx}")
            return None

if __name__ == "__main__":
    # Get the expanded path
    train_file_path = os.path.expanduser("~/Documents/SimCLR/datasets/fdd_cls_dataset/train")
    #train_annot_path = os.path.join(train_file_path, "_annotations.coco.json")

    test_file_path = os.path.expanduser("~/Documents/SimCLR/datasets/fdd_cls_dataset/test")
    #test_annot_path = os.path.join(test_file_path, "_annotations.coco.json")


    # Load the COCO dataset
    test_coco_dataset = CustomCocoClassification(root=test_file_path, transform=TransformsSimCLR(size=224).test_transform)
    train_coco_dataset = CustomCocoClassification(root=train_file_path, transform=TransformsSimCLR(size=224).test_transform)
    # Initialize a dictionary to store the class labels
    train_class_labels = set()
    test_class_labels = set()

    # # Iterate through the dataset to collect class labels
    # for idx in range(len(train_coco_dataset)):
    #     sample = train_coco_dataset[idx]
    #     #if sample is not None:
    #     imgs, target = sample
    #     x = imgs
    #     print(x.shape) # (640, 640)
    #     print(idx)
    #     for obj in target:
    #         category_id = obj['category_id']
    #         train_class_labels.add(category_id)

    for idx in range(len(train_coco_dataset)):
        sample = train_coco_dataset[idx]
        img, target = sample
        train_class_labels.add(target)

    print(len(train_class_labels))
    print(train_class_labels)

    # Print the number of unique class labels
    # print("Number of unique class labels:", len(train_class_labels))
    # print("Train_class_labels: ", train_class_labels)
    # print("Test_class_labels: ", test_class_labels)

    # print(test_class_labels.issubset(train_class_labels))

    train_loader = torch.utils.data.DataLoader(
        test_coco_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )

    print("dataset length: ", len(test_coco_dataset))

    for i, (x, y) in enumerate(train_loader):
        print(i)
        print(x.shape)
        print(torch.max(y))