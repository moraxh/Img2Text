"""
Variante 1: Fine-tuning del Encoder CNN
Permite descongelar y entrenar las capas de VGG16 después de ciertas épocas
"""

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
import torchvision.models as models

SAVE_MODEL_PATH = os.path.join("models", "caption_model_finetuned.pth")
BEST_MODEL_PATH = os.path.join("models", "best_caption_model_finetuned.pth")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations
embed_size = 256
hidden_size = 512
epochs = 100
learning_rate = 5e-4
batch_size = 32
finetune_epoch = 10  # Epoch at which to start fine-tuning encoder

class CaptionLSTMWithFineTuning(nn.Module):
  """Extended model that includes the full VGG encoder for fine-tuning"""
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
    super().__init__()
    
    # Load VGG16 as encoder
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # Remove classification layers
    self.encoder = nn.Sequential(*list(vgg.children())[:-1])
    
    # Freeze encoder initially
    for param in self.encoder.parameters():
      param.requires_grad = False
    
    # Projection layer with BatchNorm
    self.feature_proj = nn.Linear(25088, embed_size)
    self.batch_norm = nn.BatchNorm1d(embed_size)
    
    # Decoder
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.dropout = nn.Dropout(0.5)

  def forward(self, images, captions, lengths):
    """
    Forward pass with images (not pre-extracted features)
    Args:
      images: [batch_size, 3, 224, 224]
      captions: [batch_size, max_len]
      lengths: [batch_size]
    """
    # Extract features from images
    with torch.set_grad_enabled(self.encoder.training):
      features = self.encoder(images)
      features = features.view(features.size(0), -1)
    
    # Project and normalize features
    features_proj = self.feature_proj(features)
    features_proj = self.batch_norm(features_proj)
    
    # Embed captions
    embeddings = self.embed(captions)
    
    # Concatenate features as first input
    embeddings = torch.cat((features_proj.unsqueeze(1), embeddings), 1)
    
    # Adjust lengths
    lengths = lengths + 1
    
    # Pack sequences
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True)
    hiddens, _ = self.lstm(packed)
    
    # Unpack sequences
    hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
    
    # Apply dropout and linear layer
    outputs = self.linear(self.dropout(hiddens))
    
    return outputs
  
  def unfreeze_encoder(self, layers_to_unfreeze=3):
    """
    Unfreeze the last N convolutional blocks of VGG16
    Args:
      layers_to_unfreeze: Number of blocks to unfreeze from the end
    """
    # VGG16 has 5 convolutional blocks, unfreeze the last N
    vgg_features = self.encoder[0]  # VGG features module
    
    # Get all children (convolutional blocks + pooling)
    children = list(vgg_features.children())
    
    # Find convolutional blocks (MaxPool separates them)
    total_blocks = sum(1 for child in children if isinstance(child, nn.MaxPool2d))
    blocks_to_freeze = total_blocks - layers_to_unfreeze
    
    # Unfreeze parameters
    current_block = 0
    for child in children:
      if isinstance(child, nn.MaxPool2d):
        current_block += 1
      
      if current_block >= blocks_to_freeze:
        for param in child.parameters():
          param.requires_grad = True
    
    print(f"✓ Unfroze last {layers_to_unfreeze} convolutional blocks of VGG16")

def calculate_perplexity(loss):
  return math.exp(loss)

def validate_model(model, val_loader, criterion, word2idx, device):
  model.eval()
  total_loss = 0
  num_batches = 0
  
  with torch.no_grad():
    for images, targets, lengths in val_loader:
      images = images.to(device)
      targets = targets.to(device)
      lengths = lengths.to(device)
      
      outputs = model(images, targets[:, :-1], lengths - 1)
      
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

def train_model_with_finetuning(word2idx):
  """Train model with encoder fine-tuning after certain epochs"""
  vocab_size = len(word2idx)
  model = CaptionLSTMWithFineTuning(embed_size, hidden_size, vocab_size)
  model = model.to(device)
  
  if os.path.exists(SAVE_MODEL_PATH):
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    print("Fine-tuned model loaded from", SAVE_MODEL_PATH)
    return model

  model.train()
  
  # Get COCO datasets with transforms (we need images, not pre-extracted features)
  train_dataset, val_dataset = get_coco_datasets(train_transform, val_transform)
  
  # Custom collate function
  train_collate = partial(collate_fn, word2idx=word2idx)
  
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, collate_fn=train_collate)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=train_collate)
  
  criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
  
  # Initially only train decoder parameters
  decoder_params = [
    p for n, p in model.named_parameters() 
    if 'encoder' not in n and p.requires_grad
  ]
  optimizer = torch.optim.Adam(decoder_params, lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
  
  best_val_loss = float('inf')
  patience_counter = 0
  early_stopping_patience = 10
  encoder_unfrozen = False

  print(f"\nStarting training with Fine-tuning (will unfreeze encoder at epoch {finetune_epoch})")
  print(f"Training set: {len(train_dataset)} samples")
  print(f"Validation set: {len(val_dataset)} samples\n")

  for epoch in range(epochs):
    # Unfreeze encoder after specified epoch
    if epoch == finetune_epoch and not encoder_unfrozen:
      print(f"\n{'='*80}")
      print(f"EPOCH {epoch+1}: Starting Fine-Tuning of Encoder")
      print(f"{'='*80}\n")
      
      model.unfreeze_encoder(layers_to_unfreeze=2)  # Unfreeze last 2 blocks
      
      # Update optimizer to include encoder parameters
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10)  # Lower LR for fine-tuning
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
      encoder_unfrozen = True
    
    epoch_loss = 0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for images, targets, lengths in progress_bar:
      images = images.to(device)
      targets = targets.to(device)
      lengths = lengths.to(device)
      
      # Forward pass
      outputs = model(images, targets[:, :-1], lengths - 1)
      
      # Calculate loss
      loss = criterion(
        outputs[:, 1:, :].reshape(-1, vocab_size),
        targets[:, 1:].reshape(-1)
      )
      
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      
      epoch_loss += loss.item()
      num_batches += 1
      progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'finetuned': encoder_unfrozen})
  
    avg_train_loss = epoch_loss / num_batches
    train_perplexity = calculate_perplexity(avg_train_loss)
    
    # Validation
    val_loss, val_perplexity = validate_model(model, val_loader, criterion, word2idx, device)
    
    print(f"\nEpoch [{epoch+1}/{epochs}] {'(Fine-tuning)' if encoder_unfrozen else ''}")
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

if __name__ == "__main__":
  # Load vocabulary
  vocab_file = "vocabulary.pt"
  if os.path.exists(vocab_file):
    vocab = torch.load(vocab_file)
    word2idx = vocab['word2idx']
    print(f"Vocabulary loaded (size: {len(word2idx)})")
  else:
    print("Error: Vocabulary not found. Please train base model first.")
    exit(1)
  
  # Train model with fine-tuning
  train_model_with_finetuning(word2idx)
