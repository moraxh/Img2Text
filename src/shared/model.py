import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CaptionLSTM(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, feature_size=25088):
    super().__init__()
    # Encoder: Project VGG features to embed_size with BatchNorm
    self.feature_proj = nn.Linear(feature_size, embed_size)
    self.batch_norm = nn.BatchNorm1d(embed_size)
    
    # Decoder
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.dropout = nn.Dropout(0.5)

  def forward(self, features, captions, lengths):
    """
    Forward pass with packed sequences
    Args:
      features: [batch_size, feature_size]
      captions: [batch_size, max_len]
      lengths: [batch_size] - actual lengths of captions
    """
    # Project and normalize features
    features_proj = self.feature_proj(features)  # [batch, embed_size]
    features_proj = self.batch_norm(features_proj)  # BatchNorm
    
    # Embed captions
    embeddings = self.embed(captions)  # [batch, seq_len, embed_size]
    
    # Concatenate features as first input
    embeddings = torch.cat((features_proj.unsqueeze(1), embeddings), 1)
    
    # Adjust lengths to account for the prepended feature
    lengths = lengths + 1
    
    # Pack sequences for efficient LSTM processing
    packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True)
    hiddens, _ = self.lstm(packed)
    
    # Unpack sequences
    hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
    
    # Apply dropout and linear layer
    outputs = self.linear(self.dropout(hiddens))
    
    return outputs
