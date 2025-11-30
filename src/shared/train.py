import os
import torch
import torch.nn as nn
from tqdm import tqdm
from shared.utils import build_vocab
from shared.model import CaptionLSTM
from shared.dataset import PreExtractedFeaturesDataset, captions_file
from torch.utils.data import DataLoader

SAVE_MODEL_PATH = os.path.join("models", "caption_model.pth")

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

def train_model(word2idx):
  # Get model if exists
  vocab_size = len(word2idx)
  model = CaptionLSTM(embed_size, hidden_size, vocab_size)
  model = model.to(device)  # Move model to GPU
  
  if os.path.exists(SAVE_MODEL_PATH):
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    print("Model loaded from", SAVE_MODEL_PATH)
    return model

  model.train()
  
  # Load pre-extracted features dataset
  features_path = "features.pt"
  if not os.path.exists(features_path):
    raise FileNotFoundError(f"Features file not found: {features_path}. Please run extract_features.py first.")
  
  train_dataset = PreExtractedFeaturesDataset(features_path, captions_file)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
  
  criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> token
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
  
  best_loss = float('inf')

  for epoch in range(epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for features, captions in progress_bar:
      # Move tensors to GPU
      features = features.to(device)
      
      # Tokenize captions
      from shared.utils import tokenize_caption
      caption_tokens = []
      for caption in captions:
        tokens = [word2idx.get(word, word2idx['<pad>']) for word in tokenize_caption(caption)]
        tokens = [word2idx['<start>']] + tokens + [word2idx['<end>']]
        caption_tokens.append(tokens)
      
      # Pad sequences
      max_len = max(len(tokens) for tokens in caption_tokens)
      padded_tokens = []
      for tokens in caption_tokens:
        padded = tokens + [word2idx['<pad>']] * (max_len - len(tokens))
        padded_tokens.append(padded)
      
      targets = torch.tensor(padded_tokens).to(device)
      
      # Model output includes the feature token at position 0, so we skip it
      outputs = model(features, targets[:,:-1])
      loss = criterion(outputs[:,1:,:].reshape(-1, vocab_size), targets[:,1:].reshape(-1))
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item()
      progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
  
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    # Step the scheduler
    scheduler.step(avg_loss)
    
    # Save best model
    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), SAVE_MODEL_PATH)
      print(f"✓ Best model saved (loss: {best_loss:.4f})")
  
  print(f"\nTraining complete! Best loss: {best_loss:.4f}")
  return model
