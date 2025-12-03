import torch
import nltk
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (torch.cuda.is_available()):
  print("Using GPU")
else: 
  print("Using CPU")

nltk.download('punkt')

# Load pretrained VGG16 model
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg.eval()  # Modo evaluación
vgg = vgg.to(device)  # Move to GPU

# Delete the last layer (classification) to use only as feature extractor
modules = list(vgg.children())[:-1]
vgg_features = torch.nn.Sequential(*modules)

# Training transformations (with data augmentation)
train_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.RandomCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/Inference transformations (deterministic)
val_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def extract_features(image_path, transform=None):
  """Extract features from an image using VGG16"""
  if transform is None:
    transform = val_transform
    
  img = Image.open(image_path).convert('RGB')
  img_t = transform(img)
  batch_t = torch.unsqueeze(img_t, 0).to(device)  # Add batch dimension and move to GPU

  with torch.no_grad():
    features = vgg_features(batch_t)
    features = features.view(features.size(0), -1)  # Flatten

  return features.cpu()  # Return to CPU for saving

def tokenize_caption(caption):
  return nltk.tokenize.word_tokenize(caption.lower())

def caption_to_indices(caption, word2idx):
  """Convert caption to list of indices, using <unk> for unknown words"""
  tokens = tokenize_caption(caption)
  indices = [word2idx.get(word, word2idx['<unk>']) for word in tokens]
  return [word2idx['<start>']] + indices + [word2idx['<end>']]

def collate_fn(batch, word2idx):
  """Custom collate function for DataLoader to handle variable length sequences"""
  # Sort batch by caption length (descending) - required for pack_padded_sequence
  batch.sort(key=lambda x: len(tokenize_caption(x[1])), reverse=True)
  
  features, captions = zip(*batch)
  
  # Stack features
  features = torch.stack(features, 0)
  
  # Convert captions to indices
  caption_indices = [caption_to_indices(cap, word2idx) for cap in captions]
  lengths = [len(cap) for cap in caption_indices]
  
  # Pad sequences
  max_len = max(lengths)
  padded_captions = []
  for cap in caption_indices:
    padded = cap + [word2idx['<pad>']] * (max_len - len(cap))
    padded_captions.append(padded)
  
  targets = torch.LongTensor(padded_captions)
  lengths = torch.LongTensor(lengths)
  
  return features, targets, lengths


def build_vocab(captions, threshold=4):
  counter = Counter()
  for caption in captions:
      counter.update(tokenize_caption(caption))
  words = [word for word, cnt in counter.items() if cnt >= threshold]
  word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
  for word in words:
    if word not in word2idx:
      word2idx[word] = len(word2idx)
  idx2word = {idx: word for word, idx in word2idx.items()}
  return word2idx, idx2word

def generate_caption(model, features, word2idx, idx2word, max_len=20, device='cpu'):
  model.eval()
  result = ['<start>']
  inputs = torch.tensor([word2idx['<start>']]).to(device)
  features = features.to(device)
  
  with torch.no_grad():
    for _ in range(max_len):
      outputs = model(features, inputs.unsqueeze(0))
      predicted = outputs[0,-1,:].argmax().item()
      word = idx2word[predicted]
      if word == '<end>':
        break
      result.append(word)
      inputs = torch.cat([inputs, torch.tensor([predicted]).to(device)])
  
  return ' '.join(result[1:])