import torch.nn as nn
import torch

class CaptionLSTM(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, feature_size=25088):
    super().__init__()
    self.feature_proj = nn.Linear(feature_size, embed_size)  # Project VGG features to embed_size
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)

  def forward(self, features, captions):
    # Project features to embedding size
    features_proj = self.feature_proj(features)  # [batch, embed_size]
    embeddings = self.embed(captions)  # [batch, seq_len, embed_size]
    embeddings = torch.cat((features_proj.unsqueeze(1), embeddings), 1)  # Add features as first input
    hiddens, _ = self.lstm(embeddings)
    outputs = self.linear(hiddens)
    return outputs
