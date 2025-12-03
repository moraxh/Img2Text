"""
Script de validación externa con imágenes no-COCO
Genera captions para imágenes en la carpeta imagenes_validacion/
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.shared.utils import extract_features, generate_caption, val_transform
from src.shared.model import CaptionLSTM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
VALIDATION_DIR = "imagenes_validacion"
MODEL_PATH = "models/caption_model.pth"
VOCAB_PATH = "vocabulary.pt"
OUTPUT_DIR = "resultados_validacion"

# Model configuration
embed_size = 256
hidden_size = 512

def load_model_and_vocab():
  """Load trained model and vocabulary"""
  # Load vocabulary
  if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Vocabulary not found at {VOCAB_PATH}. Please train the model first.")
  
  vocab = torch.load(VOCAB_PATH)
  word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
  vocab_size = len(word2idx)
  
  # Load model
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
  
  model = CaptionLSTM(embed_size, hidden_size, vocab_size)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
  model = model.to(device)
  model.eval()
  
  print(f"Model loaded from {MODEL_PATH}")
  print(f"Vocabulary size: {vocab_size}")
  
  return model, word2idx, idx2word

def validate_external_images(model, word2idx, idx2word, max_images=None):
  """
  Generate captions for external validation images
  
  Args:
    model: Trained caption model
    word2idx: Word to index mapping
    idx2word: Index to word mapping
    max_images: Maximum number of images to process (None = all)
  """
  # Get all images from validation directory
  valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
  image_files = [
    f for f in os.listdir(VALIDATION_DIR) 
    if os.path.splitext(f.lower())[1] in valid_extensions
  ]
  
  if not image_files:
    print(f"\n⚠ No images found in {VALIDATION_DIR}/")
    print("Please add at least 10 images for external validation.")
    return
  
  if len(image_files) < 10:
    print(f"\n⚠ Warning: Only {len(image_files)} images found. The assignment requires at least 10 images.")
  
  if max_images:
    image_files = image_files[:max_images]
  
  print(f"\nProcessing {len(image_files)} images from {VALIDATION_DIR}/\n")
  
  # Create output directory
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  
  # Process each image
  results = []
  for img_file in image_files:
    img_path = os.path.join(VALIDATION_DIR, img_file)
    
    try:
      # Extract features
      features = extract_features(img_path, transform=val_transform)
      
      # Generate caption
      caption = generate_caption(model, features, word2idx, idx2word, max_len=20, device=device)
      
      results.append({
        'filename': img_file,
        'path': img_path,
        'caption': caption
      })
      
      print(f"✓ {img_file:30s} -> {caption}")
      
    except Exception as e:
      print(f"✗ Error processing {img_file}: {e}")
  
  # Save results to text file
  results_file = os.path.join(OUTPUT_DIR, "captions.txt")
  with open(results_file, 'w', encoding='utf-8') as f:
    f.write("External Validation Results\n")
    f.write("=" * 80 + "\n\n")
    for result in results:
      f.write(f"Image: {result['filename']}\n")
      f.write(f"Caption: {result['caption']}\n\n")
  
  print(f"\nResults saved to {results_file}")
  
  # Create visualization
  create_visualization(results)
  
  return results

def create_visualization(results, images_per_row=3):
  """Create a grid visualization of images with captions"""
  if not results:
    return
  
  n_images = len(results)
  n_rows = (n_images + images_per_row - 1) // images_per_row
  
  fig = plt.figure(figsize=(15, 5 * n_rows))
  gs = GridSpec(n_rows, images_per_row, figure=fig, hspace=0.4, wspace=0.3)
  
  for idx, result in enumerate(results):
    row = idx // images_per_row
    col = idx % images_per_row
    
    ax = fig.add_subplot(gs[row, col])
    
    # Load and display image
    img = Image.open(result['path']).convert('RGB')
    ax.imshow(img)
    ax.axis('off')
    
    # Add caption as title
    caption_wrapped = '\n'.join([
      result['caption'][i:i+40] 
      for i in range(0, len(result['caption']), 40)
    ])
    ax.set_title(caption_wrapped, fontsize=10, wrap=True, pad=10)
  
  plt.suptitle('External Validation Results - Generated Captions', 
               fontsize=16, fontweight='bold', y=0.995)
  
  # Save figure
  output_path = os.path.join(OUTPUT_DIR, "validation_results.png")
  plt.savefig(output_path, dpi=150, bbox_inches='tight')
  print(f"Visualization saved to {output_path}")
  
  plt.close()

def main():
  """Main function"""
  print("=" * 80)
  print("External Validation - Image Captioning")
  print("=" * 80)
  
  # Check if validation directory exists and has images
  if not os.path.exists(VALIDATION_DIR):
    print(f"\n⚠ Error: Directory {VALIDATION_DIR}/ not found!")
    print("Please create it and add at least 10 images for validation.")
    return
  
  # Load model and vocabulary
  model, word2idx, idx2word = load_model_and_vocab()
  
  # Run validation
  results = validate_external_images(model, word2idx, idx2word)
  
  if results:
    print("\n" + "=" * 80)
    print(f"✓ Validation complete! Processed {len(results)} images.")
    print(f"✓ Results saved in {OUTPUT_DIR}/")
    print("=" * 80)

if __name__ == "__main__":
  main()
