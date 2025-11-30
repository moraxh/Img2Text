from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub
import os
import torch

path = kagglehub.dataset_download("adityajn105/flickr8k")

# Dataloader
class Flickr8kDataset(Dataset):
  def __init__(self, img_dir, captions_file, transform=None):
    self.img_dir = img_dir
    self.captions_file = captions_file
    self.transform = transform
    self.image_paths = []
    self.captions = []

    with open(captions_file, 'r') as f:
      lines = f.readlines()[1:]  # Skip header
      for line in lines:
        img_name, caption = line.strip().split(',', 1)  # CSV format, split only on first comma
        self.image_paths.append(os.path.join(img_dir, img_name))
        self.captions.append(caption.strip())

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    caption = self.captions[idx]
    image = Image.open(img_path).convert('RGB')

    if self.transform:
      image = self.transform(image)

    return image, caption

data_dir = os.path.join(path, 'Images')
captions_file = os.path.join(path, 'captions.txt')
dataset = Flickr8kDataset(img_dir=data_dir, captions_file=captions_file)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Dataset for pre-extracted features
class PreExtractedFeaturesDataset(Dataset):
  def __init__(self, features_file, captions_file):
    self.features_dict = torch.load(features_file)
    self.image_names = []
    self.captions = []
    
    with open(captions_file, 'r') as f:
      lines = f.readlines()[1:]  # Skip header
      for line in lines:
        img_name, caption = line.strip().split(',', 1)
        self.image_names.append(img_name)
        self.captions.append(caption.strip())
  
  def __len__(self):
    return len(self.captions)
  
  def __getitem__(self, idx):
    img_name = self.image_names[idx]
    features = self.features_dict[img_name]
    caption = self.captions[idx]
    return features, caption