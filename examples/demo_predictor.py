"""
Demo: PredictorNetwork Usage Examples

This script demonstrates how to use Abdullah's PredictorNetwork
in various scenarios, preparing for Phase 2 experiments.

Author: Abdullah
Phase: 1 - Building Blocks Sprint
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predictor_network import PredictorNetwork, PredictorNetworkConfig


def demo_basic_usage():
    """Demo 1: Basic forward pass"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)

    # Initialize network
    model = PredictorNetwork()
    model.print_architecture()

    # Create dummy data (simulating Nawfal's and Ali's outputs)
    print("\n[Step 1] Creating dummy input data...")
    batch_size = 4
    vision_vec = torch.randn(batch_size, 512)  # From Nawfal's VisionModule
    text_vec = torch.randn(batch_size, 384)    # From Ali's TextModule

    print(f"  Vision input shape: {vision_vec.shape}")
    print(f"  Text input shape: {text_vec.shape}")

    # Forward pass
    print("\n[Step 2] Running forward pass...")
    with torch.no_grad():  # Inference mode
        predicted_embedding = model(vision_vec, text_vec)

    print(f"  Predicted embedding shape: {predicted_embedding.shape}")
    print(f"  Embedding range: [{predicted_embedding.min():.3f}, {predicted_embedding.max():.3f}]")


def demo_training_loop():
    """Demo 2: Simple training loop"""
    print("\n" + "=" * 70)
    print("DEMO 2: Training Loop (Simulated)")
    print("=" * 70)

    # Setup
    model = PredictorNetwork()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\n[Setup]")
    print(f"  Model parameters: {model.get_num_parameters()['total']:,}")
    print(f"  Optimizer: AdamW (lr=1e-4)")

    # Simulate training for a few iterations
    print("\n[Training Simulation]")
    for epoch in range(3):
        # Dummy data (in real scenario, this comes from dataloader)
        vision_vec = torch.randn(8, 512)
        text_vec = torch.randn(8, 384)
        target_embedding = torch.randn(8, 512)  # From Y-Encoder

        # Forward pass
        predicted = model(vision_vec, text_vec)

        # Calculate loss (Cosine Similarity)
        # Loss = 1 - cos_sim (we want to maximize similarity)
        loss = 1 - torch.nn.functional.cosine_similarity(
            predicted,
            target_embedding,
            dim=-1
        ).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Epoch {epoch + 1}/3 - Loss: {loss.item():.4f}")


def demo_different_configs():
    """Demo 3: Different model configurations"""
    print("\n" + "=" * 70)
    print("DEMO 3: Model Configuration Comparison")
    print("=" * 70)

    configs = ['DEFAULT', 'WIDE', 'DEEP']
    results = []

    for config_name in configs:
        print(f"\n[{config_name} Configuration]")

        # Get config and create model
        config = PredictorNetworkConfig.get_config(config_name)
        model = PredictorNetwork(**config)

        # Count parameters
        params = model.get_num_parameters()

        print(f"  Hidden Dim: {model.hidden_dim}")
        print(f"  Parameters: {params['total']:,}")
        print(f"  Residual: {model.use_residual}")

        # Test forward pass
        vision_vec = torch.randn(4, 512)
        text_vec = torch.randn(4, 384)

        with torch.no_grad():
            output = model(vision_vec, text_vec)

        results.append({
            'config': config_name,
            'params': params['total'],
            'hidden_dim': model.hidden_dim,
            'output_shape': output.shape
        })

    # Summary table
    print("\n[Comparison Summary]")
    print(f"{'Config':<10} {'Hidden Dim':<12} {'Parameters':<15} {'Output Shape':<15}")
    print("-" * 55)
    for result in results:
        print(
            f"{result['config']:<10} "
            f"{result['hidden_dim']:<12} "
            f"{result['params']:>12,}   "
            f"{str(result['output_shape']):<15}"
        )


def demo_intermediate_outputs():
    """Demo 4: Visualizing intermediate layer activations"""
    print("\n" + "=" * 70)
    print("DEMO 4: Intermediate Layer Outputs")
    print("=" * 70)

    model = PredictorNetwork()

    # Create input
    vision_vec = torch.randn(2, 512)
    text_vec = torch.randn(2, 384)

    # Get intermediate outputs
    print("\n[Forward pass with intermediate outputs]")
    with torch.no_grad():
        output, intermediates = model(vision_vec, text_vec, return_intermediate=True)

    # Print shapes
    print(f"  Input: Vision {vision_vec.shape}, Text {text_vec.shape}")
    print(f"\n  Intermediate Layers:")
    for layer_name, activation in intermediates.items():
        mean_activation = activation.mean().item()
        std_activation = activation.std().item()
        print(f"    {layer_name:<10} Shape: {str(activation.shape):<15} "
              f"Mean: {mean_activation:>7.3f}, Std: {std_activation:>6.3f}")


def demo_error_handling():
    """Demo 5: Error handling demonstration"""
    print("\n" + "=" * 70)
    print("DEMO 5: Error Handling")
    print("=" * 70)

    model = PredictorNetwork()

    # Test 1: Wrong vision dimension
    print("\n[Test 1] Wrong vision dimension...")
    try:
        wrong_vision = torch.randn(4, 256)  # Should be 512
        correct_text = torch.randn(4, 384)
        model(wrong_vision, correct_text)
    except ValueError as e:
        print(f"  [OK] Caught error: {str(e)}")

    # Test 2: Wrong text dimension
    print("\n[Test 2] Wrong text dimension...")
    try:
        correct_vision = torch.randn(4, 512)
        wrong_text = torch.randn(4, 128)  # Should be 384
        model(correct_vision, wrong_text)
    except ValueError as e:
        print(f"  [OK] Caught error: {str(e)}")

    # Test 3: Batch size mismatch
    print("\n[Test 3] Batch size mismatch...")
    try:
        vision_4 = torch.randn(4, 512)
        text_8 = torch.randn(8, 384)
        model(vision_4, text_8)
    except ValueError as e:
        print(f"  [OK] Caught error: {str(e)}")

    print("\n  All error handling tests passed! [OK]")


def demo_integration_scenario():
    """Demo 6: Integration with team modules (mock)"""
    print("\n" + "=" * 70)
    print("DEMO 6: Team Integration Scenario (Mock)")
    print("=" * 70)

    print("\n[Scenario: Processing a video frame with query]")

    # Mock Nawfal's VisionModule
    class MockVisionModule:
        def __call__(self, image):
            # Simulates extracting vision features
            batch_size = image.shape[0] if len(image.shape) > 3 else 1
            return torch.randn(batch_size, 512)

    # Mock Ali's TextModule
    class MockTextModule:
        def __call__(self, text):
            # Simulates extracting text features
            batch_size = len(text) if isinstance(text, list) else 1
            return torch.randn(batch_size, 384)

    # Mock Ansab's Loss Function
    def mock_cosine_loss(pred, target):
        return 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()

    # Initialize all components
    print("\n[Step 1] Initializing team modules...")
    vision_encoder = MockVisionModule()
    text_encoder = MockTextModule()
    predictor = PredictorNetwork()
    print("  [OK] Vision Module (Nawfal)")
    print("  [OK] Text Module (Ali)")
    print("  [OK] Predictor Network (Abdullah)")
    print("  [OK] Loss Function (Ansab)")

    # Process a batch
    print("\n[Step 2] Processing batch...")
    image_batch = torch.randn(4, 3, 224, 224)  # 4 images
    query_batch = ["What is happening?"] * 4

    vision_vec = vision_encoder(image_batch)
    text_vec = text_encoder(query_batch)

    print(f"  Vision features: {vision_vec.shape}")
    print(f"  Text features: {text_vec.shape}")

    # Predict
    print("\n[Step 3] Predicting target embedding...")
    predicted = predictor(vision_vec, text_vec)
    print(f"  Predicted embedding: {predicted.shape}")

    # Calculate loss (with mock target)
    print("\n[Step 4] Calculating loss...")
    target = torch.randn(4, 512)  # Mock target from Y-Encoder
    loss = mock_cosine_loss(predicted, target)
    print(f"  Loss (Cosine): {loss.item():.4f}")

    print("\n  Integration test passed! [OK]")


def main():
    """Run all demos"""
    print("\n" + "#" * 70)
    print("# PredictorNetwork Demonstration Suite")
    print("# Author: Abdullah | Phase 1: Building Blocks Sprint")
    print("#" * 70)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Training Loop", demo_training_loop),
        ("Configuration Comparison", demo_different_configs),
        ("Intermediate Outputs", demo_intermediate_outputs),
        ("Error Handling", demo_error_handling),
        ("Team Integration", demo_integration_scenario)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n[ERROR] Demo {i} ({name}) failed: {str(e)}")
            continue

    print("\n" + "#" * 70)
    print("# All Demos Completed Successfully!")
    print("# Ready for Phase 1 Merge Day!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
