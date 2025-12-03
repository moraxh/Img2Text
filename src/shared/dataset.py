from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import os
import torch

# MS-COCO 2014 Dataset paths (update these paths as needed)
COCO_ROOT = os.path.expanduser("~/.cache/coco")
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2014")
VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2014")
TRAIN_ANN_FILE = os.path.join(COCO_ROOT, "annotations/captions_train2014.json")
VAL_ANN_FILE = os.path.join(COCO_ROOT, "annotations/captions_val2014.json")

# COCO Dataset
class COCODataset(Dataset):
  def __init__(self, img_dir, ann_file, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.coco = COCO(ann_file)
    self.ids = list(self.coco.anns.keys())
    
  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    ann_id = self.ids[idx]
    caption = self.coco.anns[ann_id]['caption']
    img_id = self.coco.anns[ann_id]['image_id']
    img_info = self.coco.loadImgs(img_id)[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    
    image = Image.open(img_path).convert('RGB')
    
    if self.transform:
      image = self.transform(image)

    return image, caption

# Create dataset instances (will be initialized with transforms later)
train_dataset = None
val_dataset = None

def get_coco_datasets(train_transform=None, val_transform=None):
  global train_dataset, val_dataset
  train_dataset = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transform=train_transform)
  val_dataset = COCODataset(VAL_IMG_DIR, VAL_ANN_FILE, transform=val_transform)
  return train_dataset, val_dataset

# Dataset for pre-extracted features
class PreExtractedFeaturesDataset(Dataset):
  def __init__(self, features_dict, coco_dataset):
    self.features_dict = features_dict
    self.coco_dataset = coco_dataset
    self.ids = coco_dataset.ids
  
  def __len__(self):
    return len(self.ids)
  
  def __getitem__(self, idx):
    ann_id = self.ids[idx]
    caption = self.coco_dataset.coco.anns[ann_id]['caption']
    img_id = self.coco_dataset.coco.anns[ann_id]['image_id']
    img_info = self.coco_dataset.coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    
    features = self.features_dict[img_name]
    return features, caption