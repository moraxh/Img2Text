"""
BLEU Score implementation for evaluating caption quality
"""

from torchmetrics.text import BLEUScore
import torch

def calculate_bleu(predictions, references, n_gram=4):
  """
  Calculate BLEU score for image captioning
  
  Args:
    predictions: List of predicted captions (strings)
    references: List of lists of reference captions (each image can have multiple references)
    n_gram: Maximum n-gram to use (default 4 for BLEU-4)
  
  Returns:
    BLEU score as float
  """
  bleu = BLEUScore(n_gram=n_gram, smooth=True)
  score = bleu(predictions, references)
  return score.item()

def evaluate_bleu_on_dataset(model, dataloader, word2idx, idx2word, device, max_samples=1000):
  """
  Evaluate BLEU score on a dataset
  
  Args:
    model: Trained caption model
    dataloader: DataLoader for the dataset
    word2idx: Word to index mapping
    idx2word: Index to word mapping
    device: Device to run on
    max_samples: Maximum number of samples to evaluate
  
  Returns:
    Average BLEU-4 score
  """
  from shared.utils import generate_caption
  
  model.eval()
  predictions = []
  references = []
  
  count = 0
  with torch.no_grad():
    for features, captions in dataloader:
      if count >= max_samples:
        break
        
      for i in range(features.size(0)):
        if count >= max_samples:
          break
          
        # Generate caption
        feature = features[i:i+1].to(device)
        pred_caption = generate_caption(model, feature, word2idx, idx2word, max_len=20, device=device)
        
        # Get reference caption (note: COCO has multiple refs per image, here we use one)
        ref_caption = captions[i] if isinstance(captions[i], str) else ' '.join(captions[i])
        
        predictions.append(pred_caption)
        references.append([ref_caption])  # Wrap in list for torchmetrics
        
        count += 1
  
  # Calculate BLEU
  bleu_score = calculate_bleu(predictions, references)
  
  model.train()
  return bleu_score, predictions, references
