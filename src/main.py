import os
import torch
from shared.utils import extract_features, build_vocab, generate_caption
from shared.model import CaptionLSTM
from shared.train import train_model
from shared.dataset import get_coco_datasets, TRAIN_IMG_DIR, TRAIN_ANN_FILE

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations
embed_size = 256
hidden_size = 512

# Path to the image
image_path = os.path.join("media", "input.webp")

# Check if features need to be extracted
features_path = "../features.pt"  # One level up from src/
if not os.path.exists(features_path):
  print("Features not found. Running feature extraction...")
  print("This will take 2-4 hours on first run (MS-COCO is large).\n")
  print("⚠ Make sure you have downloaded MS-COCO 2014 dataset!")
  print("Expected structure:")
  print("  ~/.cache/coco/train2014/ (73,571 images)")
  print("  ~/.cache/coco/val2014/ (40,504 images)")
  print("  ~/.cache/coco/annotations/captions_train2014.json")
  print("  ~/.cache/coco/annotations/captions_val2014.json\n")
  
  # Check if COCO exists
  if not os.path.exists(os.path.expanduser("~/.cache/coco/annotations/captions_train2014.json")):
    print("❌ COCO dataset not found!")
    print("Please run: ./download_coco.sh")
    exit(1)
  
  print("✓ COCO dataset found. Starting feature extraction...")
  print("This process will run in the background. You can monitor progress.\n")
  
  os.system("python extract_features.py")
  print("\n✓ Feature extraction complete!\n")

# Build or load vocabulary
vocab_file = "../vocabulary.pt"  # One level up from src/
if os.path.exists(vocab_file):
  vocab = torch.load(vocab_file)
  word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
  vocab_size = len(word2idx)
  print(f"Vocabulary loaded from {vocab_file} (size: {vocab_size})")
else:
  print("Building vocabulary from MS-COCO captions...")
  print("This may take a few minutes...")
  
  # Load COCO dataset to get captions
  from shared.dataset import COCODataset
  train_coco = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE)
  
  captions = []
  print(f"Processing {len(train_coco.ids)} captions for vocabulary...")
  # Use all captions for vocabulary building (or a large subset)
  max_captions = min(len(train_coco.ids), 100000)
  for idx in range(max_captions):
    if idx % 10000 == 0:
      print(f"  Processed {idx}/{max_captions} captions...")
    _, caption = train_coco[idx]
    captions.append(caption)
  
  word2idx, idx2word = build_vocab(captions, threshold=4)
  vocab_size = len(word2idx)
  torch.save({'word2idx': word2idx, 'idx2word': idx2word}, vocab_file)
  print(f"✓ Vocabulary built and saved to {vocab_file} (size: {vocab_size})")


print()

# Create and train model
print("Training model...")
model = train_model(word2idx)

# Generate caption for input image
if os.path.exists(image_path):
  print(f"\nGenerating caption for {image_path}...")
  features = extract_features(image_path)
  caption = generate_caption(model, features, word2idx, idx2word, max_len=20, device=device)
  print(f"\nGenerated Caption: {caption}")
else:
  print(f"\n⚠ Input image not found at {image_path}")
  print("Please add an image to test caption generation.")

print("\n" + "="*80)
print("Training complete!")
print("\nNext steps:")
print("1. Add 10+ images to 'imagenes_validacion/' folder")
print("2. Run: python validacion_externa.py")
print("3. Try variants:")
print("   - python src/variante_finetuning.py")
print("   - python src/variante_beam_search.py")
print("="*80)

