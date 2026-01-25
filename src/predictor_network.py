"""
PredictorNetwork: Core JEPA Logic for Vision-Language Embedding Prediction

Author: Abdullah
Task: Phase 1 - Building Blocks Sprint
Description: 4-layer Neural Network that takes concatenated vision and text vectors
             and predicts the target embedding in the shared semantic space.

Architecture Design Decisions:
- Activation: GELU (used in modern transformers like BERT, Llama)
- Normalization: LayerNorm (stable for transformer-like architectures)
- Regularization: Dropout (0.1 default, can be tuned in Phase 2)
- Hidden Dimensions: Configurable, default 768 (similar to BERT-base)

References:
- VL-JEPA Paper: Uses last 8 Transformer layers of Llama-3.2-1B
- Our implementation: Simplified MLP for educational purposes
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PredictorNetwork(nn.Module):
    """
    4-Layer Multi-Layer Perceptron (MLP) Predictor for JEPA Architecture.

    This network learns the mapping: (vision_embedding, text_embedding) -> target_embedding

    Architecture Flow:
        Input: [vision_vec, text_vec] (concatenated)
             ↓
        Layer 1: Linear + LayerNorm + GELU + Dropout
             ↓
        Layer 2: Linear + LayerNorm + GELU + Dropout
             ↓
        Layer 3: Linear + LayerNorm + GELU
             ↓
        Layer 4: Linear (output projection)
             ↓
        Output: predicted_embedding

    Args:
        vision_dim (int): Dimension of vision encoder output (from Nawfal's module)
        text_dim (int): Dimension of text encoder output (from Ali's module)
        hidden_dim (int): Hidden layer dimension (default: 768)
        output_dim (int): Output embedding dimension (should match Y-Encoder space)
        dropout_rate (float): Dropout probability for regularization (default: 0.1)
        use_residual (bool): Whether to add residual connections (default: False)
    """

    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout_rate: float = 0.1,
        use_residual: bool = False
    ):
        super(PredictorNetwork, self).__init__()

        # Store configuration
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        # Calculate input dimension (concatenated vectors)
        input_dim = vision_dim + text_dim

        # Layer 1: Input -> Hidden
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Layer 2: Hidden -> Hidden
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Layer 3: Hidden -> Hidden/2 (bottleneck)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # Layer 4: Hidden/2 -> Output (no activation on final layer)
        self.layer4 = nn.Linear(hidden_dim // 2, output_dim)

        # Optional: Input projection for residual connection
        if use_residual:
            self.input_projection = nn.Linear(input_dim, output_dim)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        This helps with gradient flow in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        vision_vec: torch.Tensor,
        text_vec: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the predictor network.

        Args:
            vision_vec (torch.Tensor): Vision embeddings from X-Encoder
                                       Shape: [batch_size, vision_dim]
            text_vec (torch.Tensor): Text embeddings from query encoder
                                     Shape: [batch_size, text_dim]
            return_intermediate (bool): If True, return intermediate layer outputs
                                       for visualization/debugging

        Returns:
            torch.Tensor: Predicted target embedding
                         Shape: [batch_size, output_dim]
            OR
            Tuple[torch.Tensor, dict]: If return_intermediate=True, also returns
                                       dictionary of intermediate activations

        Raises:
            ValueError: If input tensor dimensions don't match expected sizes
        """
        # Input validation
        batch_size = vision_vec.shape[0]

        if vision_vec.shape[1] != self.vision_dim:
            raise ValueError(
                f"Vision vector dimension mismatch. Expected {self.vision_dim}, "
                f"got {vision_vec.shape[1]}"
            )

        if text_vec.shape[1] != self.text_dim:
            raise ValueError(
                f"Text vector dimension mismatch. Expected {self.text_dim}, "
                f"got {text_vec.shape[1]}"
            )

        if vision_vec.shape[0] != text_vec.shape[0]:
            raise ValueError(
                f"Batch size mismatch. Vision: {vision_vec.shape[0]}, "
                f"Text: {text_vec.shape[0]}"
            )

        # Step 1: Concatenate vision and text vectors
        # Shape: [batch_size, vision_dim + text_dim]
        combined = torch.cat([vision_vec, text_vec], dim=-1)

        # Store intermediate outputs if requested
        intermediates = {} if return_intermediate else None

        # Step 2: Pass through 4 layers
        h1 = self.layer1(combined)
        if return_intermediate:
            intermediates['layer1'] = h1

        h2 = self.layer2(h1)
        if return_intermediate:
            intermediates['layer2'] = h2

        h3 = self.layer3(h2)
        if return_intermediate:
            intermediates['layer3'] = h3

        output = self.layer4(h3)

        # Step 3: Optional residual connection
        if self.use_residual:
            residual = self.input_projection(combined)
            output = output + residual

        if return_intermediate:
            intermediates['output'] = output
            return output, intermediates

        return output

    def get_num_parameters(self) -> dict:
        """
        Calculate number of parameters in the network.

        Returns:
            dict: Dictionary with 'total', 'trainable', and 'non_trainable' counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }

    def print_architecture(self):
        """
        Print a detailed summary of the network architecture.
        Useful for debugging and understanding the model structure.
        """
        print("=" * 70)
        print("PredictorNetwork Architecture Summary")
        print("=" * 70)
        print(f"Input Configuration:")
        print(f"  - Vision Dimension: {self.vision_dim}")
        print(f"  - Text Dimension: {self.text_dim}")
        print(f"  - Combined Input: {self.vision_dim + self.text_dim}")
        print(f"\nHidden Layers:")
        print(f"  - Hidden Dimension: {self.hidden_dim}")
        print(f"  - Bottleneck Dimension: {self.hidden_dim // 2}")
        print(f"\nOutput Configuration:")
        print(f"  - Output Dimension: {self.output_dim}")
        print(f"  - Residual Connection: {self.use_residual}")
        print(f"\nParameter Count:")
        params = self.get_num_parameters()
        print(f"  - Total Parameters: {params['total']:,}")
        print(f"  - Trainable Parameters: {params['trainable']:,}")
        print("=" * 70)


class PredictorNetworkConfig:
    """
    Configuration class for PredictorNetwork.
    Makes it easy to experiment with different architectures in Phase 2.
    """

    # Default configuration (matches team's expected dimensions)
    DEFAULT = {
        'vision_dim': 512,
        'text_dim': 384,
        'hidden_dim': 768,
        'output_dim': 512,
        'dropout_rate': 0.1,
        'use_residual': False
    }

    # Wide network configuration (for Ali's Phase 2 experiment)
    WIDE = {
        'vision_dim': 512,
        'text_dim': 384,
        'hidden_dim': 1024,
        'output_dim': 512,
        'dropout_rate': 0.1,
        'use_residual': False
    }

    # Deep network configuration (for Ali's Phase 2 experiment)
    # Note: This config is for reference; actual deep version would need code changes
    DEEP = {
        'vision_dim': 512,
        'text_dim': 384,
        'hidden_dim': 768,
        'output_dim': 512,
        'dropout_rate': 0.15,  # Higher dropout for deeper network
        'use_residual': True   # Residual helps in deep networks
    }

    @classmethod
    def get_config(cls, config_name: str = 'DEFAULT') -> dict:
        """
        Get configuration by name.

        Args:
            config_name (str): One of 'DEFAULT', 'WIDE', 'DEEP'

        Returns:
            dict: Configuration dictionary
        """
        if not hasattr(cls, config_name):
            raise ValueError(f"Unknown config: {config_name}")
        return getattr(cls, config_name)


# Example usage and testing
if __name__ == "__main__":
    print("PredictorNetwork Test Suite")
    print("-" * 70)

    # Test 1: Initialize network with default config
    print("\n[Test 1] Initializing PredictorNetwork...")
    model = PredictorNetwork()
    model.print_architecture()

    # Test 2: Forward pass with dummy data
    print("\n[Test 2] Testing forward pass with dummy data...")
    batch_size = 4
    vision_dummy = torch.randn(batch_size, 512)
    text_dummy = torch.randn(batch_size, 384)

    output = model(vision_dummy, text_dummy)
    print(f"✓ Input shapes: Vision {vision_dummy.shape}, Text {text_dummy.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected output shape: [{batch_size}, 512]")

    # Test 3: Test with intermediate outputs
    print("\n[Test 3] Testing with intermediate outputs...")
    output, intermediates = model(vision_dummy, text_dummy, return_intermediate=True)
    print(f"✓ Intermediate layers captured: {list(intermediates.keys())}")
    for layer_name, activation in intermediates.items():
        print(f"  - {layer_name}: {activation.shape}")

    # Test 4: Test different configurations
    print("\n[Test 4] Testing WIDE configuration...")
    wide_config = PredictorNetworkConfig.get_config('WIDE')
    wide_model = PredictorNetwork(**wide_config)
    wide_output = wide_model(vision_dummy, text_dummy)
    print(f"✓ WIDE model output shape: {wide_output.shape}")
    print(f"✓ WIDE model parameters: {wide_model.get_num_parameters()['total']:,}")

    # Test 5: Error handling
    print("\n[Test 5] Testing error handling...")
    try:
        wrong_vision = torch.randn(batch_size, 256)  # Wrong dimension
        model(wrong_vision, text_dummy)
    except ValueError as e:
        print(f"✓ Correctly caught dimension mismatch: {str(e)}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
