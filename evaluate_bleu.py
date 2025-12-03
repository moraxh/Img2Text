"""
Script para evaluar BLEU score en el conjunto de validación
"""

import os
import torch
from shared.model import CaptionLSTM
from shared.dataset import get_coco_datasets, PreExtractedFeaturesDataset, VAL_IMG_DIR, VAL_ANN_FILE, COCODataset
from shared.bleu import evaluate_bleu_on_dataset
from torch.utils.data import DataLoader
from functools import partial
from shared.utils import collate_fn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Configurations
embed_size = 256
hidden_size = 512

def main():
    print("=" * 80)
    print("BLEU Score Evaluation on Validation Set")
    print("=" * 80)
    print()
    
    # Load vocabulary
    vocab_file = "vocabulary.pt"
    if not os.path.exists(vocab_file):
        print("❌ Error: Vocabulary not found. Please train the model first.")
        return
    
    vocab = torch.load(vocab_file)
    word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
    vocab_size = len(word2idx)
    print(f"✓ Vocabulary loaded (size: {vocab_size})")
    
    # Load model
    model_path = "models/caption_model.pth"
    if not os.path.exists(model_path):
        print("❌ Error: Model not found. Please train the model first.")
        return
    
    model = CaptionLSTM(embed_size, hidden_size, vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    print()
    
    # Load features and validation dataset
    features_path = "features.pt"
    if not os.path.exists(features_path):
        print("❌ Error: Features not found. Please run extract_features.py first.")
        return
    
    print("Loading validation dataset...")
    features_dict = torch.load(features_path)
    val_coco = COCODataset(VAL_IMG_DIR, VAL_ANN_FILE)
    val_dataset = PreExtractedFeaturesDataset(features_dict, val_coco)
    
    # Create dataloader
    val_collate = partial(collate_fn, word2idx=word2idx)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=4, collate_fn=val_collate)
    
    print(f"✓ Validation set loaded ({len(val_dataset)} samples)")
    print()
    
    # Evaluate BLEU
    print("Calculating BLEU-4 score...")
    print("(This may take several minutes)")
    print()
    
    max_samples = 1000  # Evaluar en 1000 muestras para no tardar demasiado
    bleu_score, predictions, references = evaluate_bleu_on_dataset(
        model, val_loader, word2idx, idx2word, device, max_samples=max_samples
    )
    
    print("=" * 80)
    print(f"BLEU-4 Score: {bleu_score:.4f}")
    print(f"Evaluated on: {min(max_samples, len(val_dataset))} samples")
    print("=" * 80)
    print()
    
    # Show some examples
    print("Sample predictions:")
    print("-" * 80)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Reference: {references[i][0]}")
        print(f"  Prediction: {predictions[i]}")
    print("-" * 80)
    print()
    
    # Save results
    results_file = "bleu_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("BLEU-4 Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Vocabulary size: {vocab_size}\n")
        f.write(f"Samples evaluated: {min(max_samples, len(val_dataset))}\n")
        f.write(f"BLEU-4 Score: {bleu_score:.4f}\n\n")
        f.write("Sample Predictions:\n")
        f.write("-" * 80 + "\n")
        for i in range(min(20, len(predictions))):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  Reference:  {references[i][0]}\n")
            f.write(f"  Prediction: {predictions[i]}\n")
    
    print(f"✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
