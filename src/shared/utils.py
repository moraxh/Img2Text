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

# Transformations for input images
transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
  img = Image.open(image_path).convert('RGB')
  img_t = transform(img)
  batch_t = torch.unsqueeze(img_t, 0).to(device)  # Add batch dimension and move to GPU

  with torch.no_grad():
    features = vgg_features(batch_t)
    features = features.view(features.size(0), -1)  # Flatten

  return features.cpu()  # Return to CPU for saving

def tokenize_caption(caption):
  return nltk.tokenize.word_tokenize(caption.lower())

def build_vocab(captions, threshold=1):
  counter = Counter()
  for caption in captions:
      counter.update(tokenize_caption(caption))
  words = [word for word, cnt in counter.items() if cnt >= threshold]
  word2idx = {word: idx+2 for idx, word in enumerate(words)}
  word2idx['<pad>'] = 0
  word2idx['<start>'] = 1
  word2idx['<end>'] = len(word2idx)
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