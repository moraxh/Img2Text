import os
import torch
from shared.utils import extract_features, build_vocab, generate_caption
from shared.model import CaptionLSTM
from shared.train import train_model
from shared.dataset import dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations
embed_size = 256
hidden_size = 512

# Path to the image
image_path = os.path.join("media", "input.webp")

# Check if features need to be extracted
features_path = "features.pt"
if not os.path.exists(features_path):
  print("Features not found. Running feature extraction...")
  print("This will take a few minutes on first run.\n")
  os.system("python extract_features.py")
  print("\nFeature extraction complete!\n")

# Build or load vocabulary
vocab_file = "vocabulary.pt"
if os.path.exists(vocab_file):
  vocab = torch.load(vocab_file)
  word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
  vocab_size = len(word2idx)
  print(f"Vocabulary loaded from {vocab_file} (size: {vocab_size})")
else:
  print("Building vocabulary...")
  captions = [caption for _, caption in dataset]
  word2idx, idx2word = build_vocab(captions, threshold=1)
  vocab_size = len(word2idx)
  torch.save({'word2idx': word2idx, 'idx2word': idx2word}, vocab_file)
  print(f"Vocabulary built and saved to {vocab_file} (size: {vocab_size})")
print()

# Create and train model
print("Training model...")
model = train_model(word2idx)

# Generate caption for input.jpg
print(f"\nGenerating caption for {image_path}...")
features = extract_features(image_path)
caption = generate_caption(model, features, word2idx, idx2word, max_len=20, device=device)

print(f"\nGenerated Caption: {caption}")
