import os
import torch
import torch.nn as nn
from tqdm import tqdm
from shared.utils import build_vocab, collate_fn, train_transform, val_transform
from shared.model import CaptionLSTM
from shared.dataset import get_coco_datasets, PreExtractedFeaturesDataset
from torch.utils.data import DataLoader
from functools import partial
import math

SAVE_MODEL_PATH = os.path.join("models", "caption_model.pth")
BEST_MODEL_PATH = os.path.join("models", "best_caption_model.pth")

# Check if model directory exists
if not os.path.exists("models"):
  os.makedirs("models")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configurations
embed_size = 256
hidden_size = 512
epochs = 100
learning_rate = 5e-4
batch_size = 32

def calculate_perplexity(loss):
  """Calculate perplexity from loss: PP = e^loss"""
  return math.exp(loss)

def validate_model(model, val_loader, criterion, word2idx, device):
  """Validate model and return average loss and perplexity"""
  model.eval()
  total_loss = 0
  num_batches = 0
  
  with torch.no_grad():
    for features, targets, lengths in val_loader:
      features = features.to(device)
      targets = targets.to(device)
      lengths = lengths.to(device)
      
      # Forward pass
      outputs = model(features, targets[:, :-1], lengths - 1)
      
      # Calculate loss (skip first position which is the feature projection)
      loss = criterion(
        outputs[:, 1:, :].reshape(-1, len(word2idx)),
        targets[:, 1:].reshape(-1)
      )
      
      total_loss += loss.item()
      num_batches += 1
  
  avg_loss = total_loss / num_batches
  perplexity = calculate_perplexity(avg_loss)
  
  model.train()
  return avg_loss, perplexity

def train_model(word2idx, extract_features_only=False):
  vocab_size = len(word2idx)
  model = CaptionLSTM(embed_size, hidden_size, vocab_size)
  model = model.to(device)
  
  if os.path.exists(SAVE_MODEL_PATH):
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    print("Model loaded from", SAVE_MODEL_PATH)
    return model
  
  if extract_features_only:
    return model

  model.train()
  
  # Load pre-extracted features dataset
  features_path = "../features.pt"  # One level up from src/
  if not os.path.exists(features_path):
    raise FileNotFoundError(f"Features file not found: {features_path}. Please run extract_features.py first.")
  
  # Load features and create datasets
  print("Loading features...")
  features_dict = torch.load(features_path)
  
  # Get COCO datasets (just for caption access)
  from shared.dataset import TRAIN_IMG_DIR, TRAIN_ANN_FILE, VAL_IMG_DIR, VAL_ANN_FILE, COCODataset
  train_coco = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE)
  val_coco = COCODataset(VAL_IMG_DIR, VAL_ANN_FILE)
  
  train_dataset = PreExtractedFeaturesDataset(features_dict, train_coco)
  val_dataset = PreExtractedFeaturesDataset(features_dict, val_coco)
  
  # Create custom collate function with word2idx
  train_collate = partial(collate_fn, word2idx=word2idx)
  
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, collate_fn=train_collate)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=train_collate)
  
  criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
  
  best_val_loss = float('inf')
  patience_counter = 0
  early_stopping_patience = 10

  print(f"\nStarting training on {len(train_dataset)} samples...")
  print(f"Validation set: {len(val_dataset)} samples\n")

  for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for features, targets, lengths in progress_bar:
      features = features.to(device)
      targets = targets.to(device)
      lengths = lengths.to(device)
      
      # Forward pass (exclude last token from input)
      outputs = model(features, targets[:, :-1], lengths - 1)
      
      # Calculate loss (skip first position which is the feature projection, predict from position 1 onwards)
      loss = criterion(
        outputs[:, 1:, :].reshape(-1, vocab_size),
        targets[:, 1:].reshape(-1)
      )
      
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
      optimizer.step()
      
      epoch_loss += loss.item()
      num_batches += 1
      progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
  
    avg_train_loss = epoch_loss / num_batches
    train_perplexity = calculate_perplexity(avg_train_loss)
    
    # Validation
    val_loss, val_perplexity = validate_model(model, val_loader, criterion, word2idx, device)
    
    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_perplexity:.2f}")
    print(f"  Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")
    
    # Step the scheduler
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save(model.state_dict(), BEST_MODEL_PATH)
      print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f}, Perplexity: {val_perplexity:.2f})")
    else:
      patience_counter += 1
      print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")
      
      if patience_counter >= early_stopping_patience:
        print(f"\n⚠ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
        break
  
  # Load best model
  model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
  torch.save(model.state_dict(), SAVE_MODEL_PATH)
  print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}, Perplexity: {calculate_perplexity(best_val_loss):.2f}")
  return model

