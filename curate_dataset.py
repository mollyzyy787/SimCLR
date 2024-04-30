import cv2
from pycocotools.coco import COCO
import os

#file_path = os.path.expanduser("~/Documents/SimCLR/datasets/fdd_dataset/train")
file_path = os.path.expanduser("~/Documents/SimCLR/datasets/fdd_dataset/test")
file_path = os.path.expanduser("~/Documents/SimCLR/datasets/fdd_dataset/valid")
annot_path = os.path.join(file_path, "_annotations.coco.json")

# Load COCO annotations
coco = COCO(annot_path)  # Replace 'path_to_annotations.json' with the path to your COCO annotations file
categories = coco.loadCats(coco.getCatIds())
category_mapping = {cat['id']: cat['name'] for cat in categories}

# Create output directory
output_dir = os.path.join(os.path.expanduser("~/Documents/SimCLR/datasets/fdd_cls_dataset/"), "val")
os.makedirs(output_dir, exist_ok=True)

# Iterate through annotations
for annotation in coco.dataset['annotations']:
    image_id = annotation['image_id']
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(file_path, image_info['file_name'])  # Replace 'path_to_images' with the path to your images folder
    #print(image_path)
    image = cv2.imread(image_path)
    #print(image.shape)
    
    # Get bounding box coordinates
    bbox = annotation['bbox']
    x, y, w, h = map(int, bbox)
    
    # Crop image
    cropped_image = image[y:y+h, x:x+w]
    #print(cropped_image.shape)
    # Get label
    category_id = annotation['category_id']
    label = category_mapping[category_id]

    uid = annotation['id']
    
    # Save cropped image with label
    output_filename = os.path.join(output_dir, f'image_{uid}_{category_id}.jpg')
    cv2.imwrite(output_filename, cropped_image)
