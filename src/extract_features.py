import os
import torch
from tqdm import tqdm
from shared.utils import extract_features, train_transform
from shared.dataset import get_coco_datasets, TRAIN_IMG_DIR, VAL_IMG_DIR
from PIL import Image

print("Starting feature extraction for MS-COCO 2014...")

# Get COCO datasets
train_dataset, val_dataset = get_coco_datasets(train_transform, train_transform)

print(f"Train dataset: {len(train_dataset)} captions")
print(f"Val dataset: {len(val_dataset)} captions")

# Dictionary to store features
features_dict = {}

# Extract features for training set
print("\nExtracting features from training set...")
for idx in tqdm(range(len(train_dataset)), desc="Training features"):
    ann_id = train_dataset.ids[idx]
    img_id = train_dataset.coco.anns[ann_id]['image_id']
    img_info = train_dataset.coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    
    # Only extract if we haven't seen this image before
    if img_name not in features_dict:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        try:
            features = extract_features(img_path, transform=train_transform)
            features_dict[img_name] = features.squeeze(0)  # Remove batch dimension
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Extract features for validation set
print("\nExtracting features from validation set...")
for idx in tqdm(range(len(val_dataset)), desc="Validation features"):
    ann_id = val_dataset.ids[idx]
    img_id = val_dataset.coco.anns[ann_id]['image_id']
    img_info = val_dataset.coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    
    # Only extract if we haven't seen this image before
    if img_name not in features_dict:
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        try:
            features = extract_features(img_path, transform=train_transform)
            features_dict[img_name] = features.squeeze(0)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Save features
features_path = "../features.pt"  # Save one level up from src/
torch.save(features_dict, features_path)
print(f"\n✓ Features saved to {features_path}")
print(f"✓ Total unique images: {len(features_dict)}")

