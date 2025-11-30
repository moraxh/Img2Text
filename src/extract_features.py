import os
import torch
from tqdm import tqdm
from shared.utils import extract_features
from shared.dataset import dataset

print("Starting feature extraction...")
print(f"Total images to process: {len(dataset)}")

# Dictionary to store features
features_dict = {}

# Extract features for all images
for idx, (image, caption) in enumerate(tqdm(dataset, desc="Extracting features")):
    # Get image path from dataset
    img_path = dataset.image_paths[idx]
    img_name = os.path.basename(img_path)
    
    # Only extract if we haven't seen this image before
    if img_name not in features_dict:
        features = extract_features(img_path)
        features_dict[img_name] = features.squeeze(0)  # Remove batch dimension

# Save features
features_path = "features.pt"
torch.save(features_dict, features_path)
print(f"\nFeatures saved to {features_path}")
print(f"Total unique images: {len(features_dict)}")
