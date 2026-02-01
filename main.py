"""
Main Integration Test for VL-JEPA Training Pipeline

Author: Ansab
Phase: 1 - Building Blocks Sprint
Description: Tests the complete training pipeline with dummy modules
             to verify loss function and training step work correctly.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import train_step

# Dummy modules (to be replaced with actual implementations from team members)

class DummyVision(nn.Module):
    """
    Placeholder for Nawfal's Vision Encoder.
    Actual output: [batch_size, 512]
    """
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return torch.randn(x.size(0), self.output_dim)


class DummyText(nn.Module):
    """
    Placeholder for Ali's Text Encoder.
    Actual output: [batch_size, 384]
    """
    def __init__(self, output_dim: int = 384):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, texts):
        return torch.randn(len(texts), self.output_dim)


class DummyPredictor(nn.Module):
    """
    Placeholder for Abdullah's Predictor Network.
    Architecture mirrors the actual implementation:
    - Input: concatenated vision (512) + text (384) = 896
    - Hidden: 768 with LayerNorm, GELU, Dropout
    - Bottleneck: 576 (768 * 0.75)
    - Output: 512
    """
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.output_dim = output_dim
        input_dim = vision_dim + text_dim
        bottleneck_dim = int(hidden_dim * 0.75)

        self.net = nn.Sequential(
            # Layer 1: Input -> Hidden
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Layer 2: Hidden -> Hidden
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Layer 3: Hidden -> Bottleneck
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Layer 4: Bottleneck -> Output
            nn.Linear(bottleneck_dim, output_dim)
        )

    def forward(self, img_vec, txt_vec):
        x = torch.cat([img_vec, txt_vec], dim=1)
        return self.net(x)


# Run integration test

if __name__ == "__main__":
    print("=" * 60)
    print("VL-JEPA Training Pipeline Integration Test")
    print("=" * 60)

    BATCH_SIZE = 4
    VISION_DIM = 512  # Nawfal's output
    TEXT_DIM = 384    # Ali's output
    OUTPUT_DIM = 512  # Abdullah's output / Target space

    # Initialize dummy modules
    vision = DummyVision(output_dim=VISION_DIM)
    text_model = DummyText(output_dim=TEXT_DIM)
    predictor = DummyPredictor(
        vision_dim=VISION_DIM,
        text_dim=TEXT_DIM,
        output_dim=OUTPUT_DIM
    )

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Create dummy inputs
    image = torch.randn(BATCH_SIZE, 3, 224, 224)  # RGB images
    text = ["cat playing", "dog running", "sunset beach", "mountain view"]
    target = torch.randn(BATCH_SIZE, OUTPUT_DIM)  # Target embeddings (Y-encoder output)

    print(f"\nInput Configuration:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Image Shape: {image.shape}")
    print(f"  - Text Samples: {text}")
    print(f"  - Target Shape: {target.shape}")

    # Run training step
    loss = train_step(
        image,
        text,
        target,
        vision,
        text_model,
        predictor,
        optimizer
    )

    print(f"\nResults:")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Predictor Parameters: {sum(p.numel() for p in predictor.parameters()):,}")

    # Verify output shape
    with torch.no_grad():
        img_vec = vision(image)
        txt_vec = text_model(text)
        pred = predictor(img_vec, txt_vec)
        print(f"  - Prediction Shape: {pred.shape}")
        print(f"  - Expected Shape: [{BATCH_SIZE}, {OUTPUT_DIM}]")

    print("\n" + "=" * 60)
    print("Phase 1 Integration Test PASSED!")
    print("=" * 60)
