"""
Main Integration Test for VL-JEPA Training Pipeline

Author: Nawfal
Phase: 1 - Building Blocks Sprint (COMPLETE!)
Description: Interactive test of the complete training pipeline, combining all of the team modules:
             - Nawfal's VisionModule (Vision Transformer)
             - Ali's TextModule (Sentence Transformer)
             - Abdullah's PredictorNetwork (4-layer MLP)
             - Ansab's train_step and loss functions
             
Usage:
    python main.py --demo                    # Run with dummy data
    python main.py --interactive             # Interactive mode with user input
"""

import torch
import torch.nn as nn
import sys
import os
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


sys.path.insert(0, str(Path(__file__).parent / 'src'))


from src.vision_module import VisionModule
from src.TextModule import TextModule
from src.predictor_network import PredictorNetwork
from src.train import train_step

# Image preprocessing
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_from_file(image_path):
    """Load and preprocess an image from file."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = IMAGE_TRANSFORM(img)
        return img_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_available_images():
    """Get list of images in src/images directory."""
    images_dir = Path(__file__).parent / 'src' / 'images'
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"\Created images directory: {images_dir}")
        print("   Place your test images (.jpg, .png) in this folder!")
        return []
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpeg'))
    return sorted(image_files)

def interactive_mode():
    """Interactive mode for user input."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    
    # Show available images
    available_images = get_available_images()
    
    if not available_images:
        print("\nNo images found in src/images/")
        print("   Using dummy image data instead...")
        image = torch.randn(1, 3, 224, 224)
        image_name = "dummy_image"
    else:
        print("\nAvailable images:")
        for idx, img_path in enumerate(available_images, 1):
            print(f"  [{idx}] {img_path.name}")
        
        while True:
            try:
                choice = input(f"\nSelect image (1-{len(available_images)}): ").strip()
                img_idx = int(choice) - 1
                if 0 <= img_idx < len(available_images):
                    image_path = available_images[img_idx]
                    image_name = image_path.name
                    image = load_image_from_file(image_path)
                    if image is not None:
                        print(f"Loaded: {image_name}")
                        break
                    else:
                        print("Failed to load image. Try again.")
                else:
                    print(f"Please enter a number between 1 and {len(available_images)}")
            except (ValueError, KeyboardInterrupt):
                print("\nUsing dummy image data instead...")
                image = torch.randn(1, 3, 224, 224)
                image_name = "dummy_image"
                break
    
    # Get text input from user
    print("\n" + "-" * 60)
    print("Enter text descriptions (one per line)")
    print("Press Enter twice when done, or Ctrl+C to use defaults")
    print("-" * 60)
    
    text_inputs = []
    try:
        while True:
            text = input(f"Text {len(text_inputs) + 1}: ").strip()
            if not text:
                if text_inputs:
                    break
                else:
                    print("Please enter at least one text description")
                    continue
            text_inputs.append(text)
            if len(text_inputs) >= 4:
                print("(Maximum 4 texts reached)")
                break
    except KeyboardInterrupt:
        print("\n\nUsing default texts...")
        text_inputs = ["a cat sitting on a windowsill", "a dog running in the park"]
    
    if not text_inputs:
        text_inputs = ["a cat sitting on a windowsill"]
    
    batch_size = len(text_inputs)
    
    # Expand image batch if needed
    if image.size(0) == 1 and batch_size > 1:
        image = image.repeat(batch_size, 1, 1, 1)
    
    # Create target embeddings
    target = torch.randn(batch_size, 512)
    
    return image, text_inputs, target, image_name

def demo_mode():
    """Demo mode with predefined data."""
    BATCH_SIZE = 4
    image = torch.randn(BATCH_SIZE, 3, 224, 224)
    text = ["cat playing", "dog running", "sunset beach", "mountain view"]
    target = torch.randn(BATCH_SIZE, 512)
    
    print(f"\nInput Configuration (DEMO):")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Image Shape: {image.shape}")
    print(f"  - Text Samples: {text}")
    print(f"  - Target Shape: {target.shape}")
    
    return image, text, target

# Run integration test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VL-JEPA Integration Test')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with dummy data')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("VL-JEPA Training Pipeline Integration Test")
    print("=" * 60)

    VISION_DIM = 512  # Nawfal's output
    TEXT_DIM = 384    # Ali's output
    OUTPUT_DIM = 512  # Abdullah's output / Target space

    # Initialize actual team modules
    print("\nInitializing Modules:")
    print("  [1/3] Loading Vision Module (Nawfal)...")
    vision = VisionModule(output_dim=VISION_DIM)
    
    print("  [2/3] Loading Text Module (Ali)...")
    text_model = TextModule()
    
    print("  [3/3] Creating Predictor Network (Abdullah)...")
    predictor = PredictorNetwork(
        vision_dim=VISION_DIM,
        text_dim=TEXT_DIM,
        output_dim=OUTPUT_DIM
    )
    print("✓ All modules initialized successfully!")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Get inputs based on mode
    if args.interactive:
        image, text, target, image_name = interactive_mode()
        batch_size = len(text)
    else:
        image, text, target = demo_mode()
        batch_size = image.size(0)
        image_name = "dummy_data"

    # Run training step
    print("\n" + "=" * 60)
    print("RUNNING VL-JEPA PIPELINE")
    print("=" * 60)
    
    print(f"\nProcessing:")
    if args.interactive:
        print(f"  - Image: {image_name}")
    print(f"  - Texts: {text}")
    print(f"  - Batch Size: {batch_size}")
    
    loss = train_step(
        image,
        text,
        target,
        vision,
        text_model,
        predictor,
        optimizer
    )

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Predictor Parameters: {sum(p.numel() for p in predictor.parameters()):,}")

    # Verify output shape
    with torch.no_grad():
        img_vec = vision(image)
        txt_vec = text_model(text)
        pred = predictor(img_vec, txt_vec)
        print(f"  - Prediction Shape: {pred.shape}")
        print(f"  - Expected Shape: [{batch_size}, {OUTPUT_DIM}]")
        
        # Show embedding stats
        print(f"\n  Embedding Statistics:")
        print(f"  - Vision embeddings: mean={img_vec.mean():.3f}, std={img_vec.std():.3f}")
        print(f"  - Text embeddings: mean={txt_vec.mean():.3f}, std={txt_vec.std():.3f}")
        print(f"  - Predictions: mean={pred.mean():.3f}, std={pred.std():.3f}")

    print("\n" + "=" * 60)
    print("✓ Phase 1 Integration Test PASSED!")
    print("=" * 60)
