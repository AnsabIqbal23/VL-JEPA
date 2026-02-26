import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from pathlib import Path
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from vision_module import VisionModule
from TextModule import TextModule
from predictor_network import PredictorNetwork
from loss import cosine_similarity_loss
from train import train_step, validate_step


class AugmentationConfig:
    """Configure augmentation strategies"""
    
    # Variant A: NO augmentation (baseline)
    NO_AUG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Variant B: STRONG augmentation (your experiment)
    STRONG_AUG = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # Random rotations (-15 to +15 degrees)
        transforms.RandomRotation(degrees=15),
        
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Random color jitter (brightness, contrast, saturation, hue)
        transforms.ColorJitter(
            brightness=0.2,      # ±20% brightness
            contrast=0.2,        # ±20% contrast
            saturation=0.2,      # ±20% saturation
            hue=0.1              # ±10% hue shift
        ),
        
        # Random Gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Random perspective transform (makes image look 3D)
        transforms.RandomPerspective(p=0.5, distortion_scale=0.2),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    @staticmethod
    def get_transform(variant: str):
        """Get augmentation based on variant"""
        if variant == "no_aug":
            return AugmentationConfig.NO_AUG
        elif variant == "strong_aug":
            return AugmentationConfig.STRONG_AUG
        else:
            raise ValueError(f"Unknown variant: {variant}")


class COCOImageTextDataset(Dataset):
    """
    COCO dataset wrapper for image-text pairs.
    Downloads automatically on first use.
    """
    
    def __init__(self, variant="no_aug", num_samples=1000, split='train'):
        """
        Args:
            variant: "no_aug" or "strong_aug"
            num_samples: How many image-text pairs to use (for speed)
            split: "train" or "val"
        """
        self.variant = variant
        self.num_samples = num_samples
        self.split = split
        self.transform = AugmentationConfig.get_transform(variant)
        
        # We'll use COCO Captions from torchvision
        try:
            from torchvision.datasets import CocoCaptions
            
            # Create dummy root (will download automatically)
            root = Path('/tmp/coco_data')
            root.mkdir(exist_ok=True, parents=True)
            
            # Download COCO (only captions, images downloaded on first access)
            print(f"Loading COCO dataset ({split})...")
            self.dataset = CocoCaptions(
                root=str(root),
                annFile=f'{root}/annotations/captions_{split}2014.json',
                download=True
            )
            
            # Limit to num_samples for training speed
            if len(self.dataset) > num_samples:
                indices = np.random.choice(len(self.dataset), num_samples, replace=False)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
                
        except Exception as e:
            print(f"COCO loading failed: {e}")
            print("Falling back to ImageNet-like synthetic dataset...")
            self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """Create synthetic image-text pairs if COCO unavailable"""
        self.synthetic_data = True
        self.images = torch.randn(self.num_samples, 3, 224, 224)
        
        # Text descriptions
        descriptions = [
            "a person eating food",
            "a dog running in the park",
            "a cat sitting on a chair",
            "a car driving on a street",
            "a bird flying in the sky",
            "a boat on the water",
            "a building in the city",
            "a flower in a garden",
        ]
        
        self.captions = [
            descriptions[i % len(descriptions)] 
            for i in range(self.num_samples)
        ]
    
    def __len__(self):
        if hasattr(self, 'synthetic_data'):
            return len(self.images)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if hasattr(self, 'synthetic_data'):
            image = self.images[idx]
            caption = self.captions[idx]
        else:
            image, captions = self.dataset[idx]
            # Randomly select one caption
            caption = captions[np.random.randint(len(captions))]
            image = self.transform(image)
        
        return image, caption


class VLJEPATrainer:
    """Complete training pipeline for VL-JEPA with data augmentation experiment"""
    
    def __init__(self, variant="no_aug", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            variant: "no_aug" or "strong_aug"
            device: "cuda" or "cpu"
        """
        self.variant = variant
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cosine_sim': []
        }
        
        print(f"\n{'='*70}")
        print(f"Training VL-JEPA with variant: {variant.upper()}")
        print(f"Device: {device}")
        print(f"{'='*70}")
        
        # Initialize modules
        print("\n[1/4] Initializing Vision Module...")
        self.vision_module = VisionModule(output_dim=512).to(device)
        self.vision_module.eval()  # Freeze for Phase 2
        
        print("[2/4] Initializing Text Module...")
        self.text_module = TextModule().to(device)
        self.text_module.eval()  # Freeze for Phase 2
        
        print("[3/4] Initializing Predictor Network (trainable)...")
        self.predictor = PredictorNetwork(
            vision_dim=512,
            text_dim=384,
            output_dim=512
        ).to(device)
        self.predictor.train()  # This is what we train
        
        print("[4/4] Creating optimizer...")
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)
        
        print(f"\nPredictor parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.predictor.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            # Move images to device
            images = images.to(self.device)
            
            # Handle text (list of strings)
            if isinstance(captions, (list, tuple)):
                text_list = list(captions)
            else:
                text_list = captions.tolist()
            
            # Generate target embeddings (from Y-encoder, simulated)
            target = torch.randn(images.size(0), 512).to(self.device)
            
            try:
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    img_vec = self.vision_module(images)      # [B, 512]
                    text_vec = self.text_module(text_list)    # [B, 384]
                    text_vec = text_vec.to(self.device)
                
                # Predictor forward (only this is trained)
                pred = self.predictor(img_vec, text_vec)      # [B, 512]
                
                # Compute loss
                loss = cosine_similarity_loss(pred, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate model"""
        self.predictor.eval()
        total_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0
        
        for images, captions in val_loader:
            images = images.to(self.device)
            
            if isinstance(captions, (list, tuple)):
                text_list = list(captions)
            else:
                text_list = captions.tolist()
            
            target = torch.randn(images.size(0), 512).to(self.device)
            
            try:
                # Forward pass
                img_vec = self.vision_module(images)
                text_vec = self.text_module(text_list).to(self.device)
                pred = self.predictor(img_vec, text_vec)
                
                # Metrics
                loss = cosine_similarity_loss(pred, target)
                cosine_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
                
                total_loss += loss.item()
                total_cosine_sim += cosine_sim.item()
                num_batches += 1
            
            except Exception as e:
                print(f"Validation error: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_cosine_sim = total_cosine_sim / max(num_batches, 1)
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_cosine_sim'].append(avg_cosine_sim)
        
        return avg_loss, avg_cosine_sim
    
    def train(self, num_epochs=5, batch_size=8):
        """Full training loop"""
        
        # Load dataset
        print("\nLoading dataset...")
        train_dataset = COCOImageTextDataset(
            variant=self.variant,
            num_samples=100,  # Small for Colab free tier
            split='train'
        )
        val_dataset = COCOImageTextDataset(
            variant=self.variant,
            num_samples=20,
            split='val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nTraining Configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Train Samples: {len(train_dataset)}")
        print(f"  - Val Samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_cosine_sim = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Cosine Similarity: {val_cosine_sim:.4f}")
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        torch.save({
            'model_state': self.predictor.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'variant': self.variant
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def plot_results(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Loss Curve ({self.variant})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Cosine similarity curve
        axes[1].plot(self.history['val_cosine_sim'], label='Val Cosine Sim', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title(f'Validation Cosine Similarity ({self.variant})')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    """Run the augmentation experiment"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'#'*70}")
    print("# PHASE 2: DATA AUGMENTATION EXPERIMENT FOR VL-JEPA")
    print(f"{'#'*70}")
    
    # Variant A: No augmentation (baseline)
    print("\n\n" + "="*70)
    print("VARIANT A: NO AUGMENTATION (BASELINE)")
    print("="*70)
    
    trainer_baseline = VLJEPATrainer(variant="no_aug", device=device)
    history_baseline = trainer_baseline.train(num_epochs=3, batch_size=8)
    trainer_baseline.save_checkpoint('/tmp/vl_jepa_baseline.pth')
    trainer_baseline.plot_results('/tmp/results_baseline.png')
    
    # Variant B: Strong augmentation (your experiment)
    print("\n\n" + "="*70)
    print("VARIANT B: STRONG AUGMENTATION (YOUR EXPERIMENT)")
    print("="*70)
    
    trainer_augmented = VLJEPATrainer(variant="strong_aug", device=device)
    history_augmented = trainer_augmented.train(num_epochs=3, batch_size=8)
    trainer_augmented.save_checkpoint('/tmp/vl_jepa_augmented.pth')
    trainer_augmented.plot_results('/tmp/results_augmented.png')
    
    # Compare results
    print("\n\n" + "="*70)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*70)
    
    print(f"\nFinal Metrics:")
    print(f"{'Metric':<30} {'No Aug':<15} {'Strong Aug':<15}")
    print("-" * 60)
    
    final_val_loss_baseline = history_baseline['val_loss'][-1]
    final_val_loss_augmented = history_augmented['val_loss'][-1]
    
    final_cosine_baseline = history_baseline['val_cosine_sim'][-1]
    final_cosine_augmented = history_augmented['val_cosine_sim'][-1]
    
    print(f"{'Final Val Loss':<30} {final_val_loss_baseline:<15.4f} {final_val_loss_augmented:<15.4f}")
    print(f"{'Final Cosine Similarity':<30} {final_cosine_baseline:<15.4f} {final_cosine_augmented:<15.4f}")
    
    improvement = ((final_cosine_baseline - final_cosine_augmented) / final_cosine_baseline) * 100
    print(f"\nCosine Similarity Improvement: {improvement:.2f}%")
    
    if final_cosine_augmented > final_cosine_baseline:
        print("\n✓ DATA AUGMENTATION IMPROVED MODEL ROBUSTNESS!")
    else:
        print("\n✗ Data augmentation did not improve performance in this run.")
        print("  (This can happen with small datasets. Try larger num_samples.)")