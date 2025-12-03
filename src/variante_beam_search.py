"""
Variante 2: Beam Search para generación de captions
Mejora la generación de texto explorando múltiples hipótesis simultáneamente
"""

import torch
import torch.nn.functional as F
from shared.utils import extract_features, val_transform
from shared.model import CaptionLSTM

def beam_search_caption(model, features, word2idx, idx2word, beam_width=5, max_len=20, device='cpu'):
  """
  Generate caption using Beam Search
  
  Args:
    model: Trained caption model
    features: Image features [1, feature_size]
    word2idx: Word to index mapping
    idx2word: Index to word mapping
    beam_width: Number of beams to maintain (k)
    max_len: Maximum caption length
    device: Device to run on
    
  Returns:
    Best caption as string
  """
  model.eval()
  features = features.to(device)
  
  # Initialize beams: each beam is (sequence, score)
  # Start with <start> token
  start_idx = word2idx['<start>']
  end_idx = word2idx['<end>']
  
  # Initial beam: ([<start>], 0.0)
  beams = [([start_idx], 0.0)]
  completed_beams = []
  
  with torch.no_grad():
    for step in range(max_len):
      all_candidates = []
      
      for seq, score in beams:
        # If sequence already ended, add to completed
        if seq[-1] == end_idx:
          completed_beams.append((seq, score))
          continue
        
        # Prepare input
        inputs = torch.LongTensor(seq).unsqueeze(0).to(device)
        lengths = torch.LongTensor([len(seq)]).to(device)
        
        # Get predictions
        outputs = model(features, inputs, lengths)
        
        # Get log probabilities for last position
        logits = outputs[0, -1, :]
        log_probs = F.log_softmax(logits, dim=0)
        
        # Get top k candidates
        top_log_probs, top_indices = torch.topk(log_probs, beam_width)
        
        # Create new candidates
        for i in range(beam_width):
          next_word = top_indices[i].item()
          next_log_prob = top_log_probs[i].item()
          
          new_seq = seq + [next_word]
          new_score = score + next_log_prob
          
          all_candidates.append((new_seq, new_score))
      
      # If no candidates, break
      if not all_candidates:
        break
      
      # Sort candidates by score and keep top k
      all_candidates.sort(key=lambda x: x[1], reverse=True)
      beams = all_candidates[:beam_width]
      
      # If all beams have ended, break
      if all(seq[-1] == end_idx for seq, _ in beams):
        completed_beams.extend(beams)
        break
    
    # Add remaining beams to completed
    completed_beams.extend(beams)
    
    # Select best beam
    if not completed_beams:
      completed_beams = beams
    
    best_seq, best_score = max(completed_beams, key=lambda x: x[1] / len(x[0]))  # Normalize by length
    
    # Convert to words
    caption_words = []
    for idx in best_seq[1:]:  # Skip <start>
      if idx == end_idx:
        break
      word = idx2word.get(idx, '<unk>')
      caption_words.append(word)
    
    return ' '.join(caption_words)

def greedy_caption(model, features, word2idx, idx2word, max_len=20, device='cpu'):
  """
  Generate caption using greedy search (baseline for comparison)
  
  Args:
    model: Trained caption model
    features: Image features
    word2idx: Word to index mapping
    idx2word: Index to word mapping
    max_len: Maximum caption length
    device: Device to run on
    
  Returns:
    Caption as string
  """
  model.eval()
  features = features.to(device)
  
  result = [word2idx['<start>']]
  
  with torch.no_grad():
    for _ in range(max_len):
      inputs = torch.LongTensor(result).unsqueeze(0).to(device)
      lengths = torch.LongTensor([len(result)]).to(device)
      
      outputs = model(features, inputs, lengths)
      predicted = outputs[0, -1, :].argmax().item()
      
      result.append(predicted)
      
      if predicted == word2idx['<end>']:
        break
  
  # Convert to words
  caption_words = []
  for idx in result[1:]:  # Skip <start>
    if idx == word2idx['<end>']:
      break
    word = idx2word.get(idx, '<unk>')
    caption_words.append(word)
  
  return ' '.join(caption_words)

def compare_search_methods(model, image_path, word2idx, idx2word, beam_widths=[3, 5, 10], device='cpu'):
  """
  Compare greedy search vs beam search with different beam widths
  
  Args:
    model: Trained caption model
    image_path: Path to image
    word2idx: Word to index mapping
    idx2word: Index to word mapping
    beam_widths: List of beam widths to try
    device: Device to run on
    
  Returns:
    Dictionary with results from different methods
  """
  # Extract features
  features = extract_features(image_path, transform=val_transform)
  
  results = {}
  
  # Greedy search
  greedy_result = greedy_caption(model, features, word2idx, idx2word, max_len=20, device=device)
  results['greedy'] = greedy_result
  
  # Beam search with different widths
  for width in beam_widths:
    beam_result = beam_search_caption(model, features, word2idx, idx2word, beam_width=width, max_len=20, device=device)
    results[f'beam_{width}'] = beam_result
  
  return results

if __name__ == "__main__":
  import os
  
  # Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}\n")
  
  # Load vocabulary
  vocab_file = "vocabulary.pt"
  if not os.path.exists(vocab_file):
    print("Error: Vocabulary not found. Please train the model first.")
    exit(1)
  
  vocab = torch.load(vocab_file)
  word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
  vocab_size = len(word2idx)
  
  # Load model
  model_path = "models/caption_model.pth"
  if not os.path.exists(model_path):
    print("Error: Model not found. Please train the model first.")
    exit(1)
  
  model = CaptionLSTM(256, 512, vocab_size)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model = model.to(device)
  model.eval()
  
  print(f"Model loaded from {model_path}")
  print(f"Vocabulary size: {vocab_size}\n")
  
  # Test on sample image
  image_path = os.path.join("media", "input.webp")
  
  if not os.path.exists(image_path):
    print(f"Error: Test image not found at {image_path}")
    exit(1)
  
  print(f"Comparing search methods on: {image_path}\n")
  print("=" * 80)
  
  results = compare_search_methods(model, image_path, word2idx, idx2word, beam_widths=[3, 5, 10], device=device)
  
  # Display results
  for method, caption in results.items():
    print(f"{method.upper():15s}: {caption}")
  
  print("=" * 80)
  
  print("\n✓ Beam Search provides better global coherence compared to greedy search")
  print("✓ Larger beam widths explore more hypotheses but are computationally expensive")
