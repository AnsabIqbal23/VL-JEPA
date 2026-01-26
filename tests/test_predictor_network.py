"""
Unit Tests for PredictorNetwork

Author: Abdullah
Phase: 1 - Building Blocks Sprint

Test Coverage:
1. Initialization tests
2. Forward pass tests
3. Shape validation tests
4. Error handling tests
5. Integration tests with different configurations
6. Parameter counting tests
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predictor_network import PredictorNetwork, PredictorNetworkConfig


class TestPredictorNetworkInitialization:
    """Test suite for network initialization"""

    def test_default_initialization(self):
        """Test that network initializes with default parameters"""
        model = PredictorNetwork()
        assert model.vision_dim == 512
        assert model.text_dim == 384
        assert model.hidden_dim == 768
        assert model.output_dim == 512

    def test_custom_initialization(self):
        """Test network with custom dimensions"""
        model = PredictorNetwork(
            vision_dim=256,
            text_dim=128,
            hidden_dim=512,
            output_dim=256
        )
        assert model.vision_dim == 256
        assert model.text_dim == 128
        assert model.hidden_dim == 512
        assert model.output_dim == 256

    def test_residual_initialization(self):
        """Test that residual connection is properly initialized"""
        model = PredictorNetwork(use_residual=True)
        assert hasattr(model, 'input_projection')
        assert isinstance(model.input_projection, torch.nn.Linear)


class TestPredictorNetworkForward:
    """Test suite for forward pass functionality"""

    @pytest.fixture
    def model(self):
        """Fixture: Standard model for testing"""
        return PredictorNetwork()

    @pytest.fixture
    def dummy_data(self):
        """Fixture: Dummy input data"""
        batch_size = 4
        vision_vec = torch.randn(batch_size, 512)
        text_vec = torch.randn(batch_size, 384)
        return vision_vec, text_vec

    def test_forward_output_shape(self, model, dummy_data):
        """Test that forward pass produces correct output shape"""
        vision_vec, text_vec = dummy_data
        output = model(vision_vec, text_vec)

        expected_shape = (4, 512)  # (batch_size, output_dim)
        assert output.shape == expected_shape

    def test_forward_no_nans(self, model, dummy_data):
        """Test that forward pass doesn't produce NaN values"""
        vision_vec, text_vec = dummy_data
        output = model(vision_vec, text_vec)

        assert not torch.isnan(output).any()

    def test_forward_different_batch_sizes(self, model):
        """Test forward pass with different batch sizes"""
        for batch_size in [1, 2, 8, 16, 32]:
            vision_vec = torch.randn(batch_size, 512)
            text_vec = torch.randn(batch_size, 384)
            output = model(vision_vec, text_vec)

            assert output.shape == (batch_size, 512)

    def test_forward_with_intermediates(self, model, dummy_data):
        """Test that intermediate outputs are returned correctly"""
        vision_vec, text_vec = dummy_data
        output, intermediates = model(vision_vec, text_vec, return_intermediate=True)

        # Check that all expected layers are present
        expected_keys = ['layer1', 'layer2', 'layer3', 'output']
        assert all(key in intermediates for key in expected_keys)

        # Check shapes
        batch_size = vision_vec.shape[0]
        assert intermediates['layer1'].shape == (batch_size, 768)  # hidden_dim
        assert intermediates['layer2'].shape == (batch_size, 768)
        assert intermediates['layer3'].shape == (batch_size, 384)  # hidden_dim // 2
        assert intermediates['output'].shape == (batch_size, 512)  # output_dim


class TestPredictorNetworkErrorHandling:
    """Test suite for error handling"""

    @pytest.fixture
    def model(self):
        return PredictorNetwork()

    def test_wrong_vision_dimension(self, model):
        """Test that wrong vision dimension raises error"""
        vision_vec = torch.randn(4, 256)  # Wrong: should be 512
        text_vec = torch.randn(4, 384)

        with pytest.raises(ValueError, match="Vision vector dimension mismatch"):
            model(vision_vec, text_vec)

    def test_wrong_text_dimension(self, model):
        """Test that wrong text dimension raises error"""
        vision_vec = torch.randn(4, 512)
        text_vec = torch.randn(4, 128)  # Wrong: should be 384

        with pytest.raises(ValueError, match="Text vector dimension mismatch"):
            model(vision_vec, text_vec)

    def test_batch_size_mismatch(self, model):
        """Test that mismatched batch sizes raise error"""
        vision_vec = torch.randn(4, 512)
        text_vec = torch.randn(8, 384)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(vision_vec, text_vec)


class TestPredictorNetworkGradients:
    """Test suite for gradient flow"""

    def test_gradients_flow(self):
        """Test that gradients flow through the network"""
        model = PredictorNetwork()
        model.train()

        vision_vec = torch.randn(4, 512, requires_grad=True)
        text_vec = torch.randn(4, 384, requires_grad=True)

        output = model(vision_vec, text_vec)
        loss = output.mean()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_no_gradient_in_eval_mode(self):
        """Test that no gradients are computed in eval mode"""
        model = PredictorNetwork()
        model.eval()

        with torch.no_grad():
            vision_vec = torch.randn(4, 512)
            text_vec = torch.randn(4, 384)
            output = model(vision_vec, text_vec)

        assert not output.requires_grad


class TestPredictorNetworkConfigurations:
    """Test suite for different model configurations"""

    def test_default_config(self):
        """Test DEFAULT configuration"""
        config = PredictorNetworkConfig.get_config('DEFAULT')
        model = PredictorNetwork(**config)

        vision_vec = torch.randn(2, 512)
        text_vec = torch.randn(2, 384)
        output = model(vision_vec, text_vec)

        assert output.shape == (2, 512)

    def test_wide_config(self):
        """Test WIDE configuration"""
        config = PredictorNetworkConfig.get_config('WIDE')
        model = PredictorNetwork(**config)

        vision_vec = torch.randn(2, 512)
        text_vec = torch.randn(2, 384)
        output = model(vision_vec, text_vec)

        assert output.shape == (2, 512)
        # WIDE config should have more parameters
        assert model.get_num_parameters()['total'] > 500_000

    def test_deep_config(self):
        """Test DEEP configuration"""
        config = PredictorNetworkConfig.get_config('DEEP')
        model = PredictorNetwork(**config)

        vision_vec = torch.randn(2, 512)
        text_vec = torch.randn(2, 384)
        output = model(vision_vec, text_vec)

        assert output.shape == (2, 512)


class TestPredictorNetworkParameters:
    """Test suite for parameter counting"""

    def test_parameter_count(self):
        """Test that parameter counting is accurate"""
        model = PredictorNetwork()
        params = model.get_num_parameters()

        # Check that all keys exist
        assert 'total' in params
        assert 'trainable' in params
        assert 'non_trainable' in params

        # Check that counts make sense
        assert params['total'] > 0
        assert params['trainable'] == params['total']  # All params trainable by default
        assert params['non_trainable'] == 0

    def test_parameter_count_different_sizes(self):
        """Test parameter counts for different model sizes"""
        small_model = PredictorNetwork(hidden_dim=256)
        large_model = PredictorNetwork(hidden_dim=1024)

        small_params = small_model.get_num_parameters()['total']
        large_params = large_model.get_num_parameters()['total']

        # Larger model should have more parameters
        assert large_params > small_params


class TestPredictorNetworkIntegration:
    """Integration tests simulating real usage"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: initialize -> forward -> loss"""
        # Initialize model
        model = PredictorNetwork()
        model.train()

        # Create dummy data
        batch_size = 8
        vision_vec = torch.randn(batch_size, 512)
        text_vec = torch.randn(batch_size, 384)
        target_embedding = torch.randn(batch_size, 512)

        # Forward pass
        predicted_embedding = model(vision_vec, text_vec)

        # Calculate loss (Cosine Similarity as mentioned in paper)
        loss = 1 - torch.nn.functional.cosine_similarity(
            predicted_embedding,
            target_embedding,
            dim=-1
        ).mean()

        # Backward pass
        loss.backward()

        # Check that loss is valid
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_inference_mode(self):
        """Test model in inference mode"""
        model = PredictorNetwork()
        model.eval()

        with torch.no_grad():
            vision_vec = torch.randn(1, 512)
            text_vec = torch.randn(1, 384)
            output = model(vision_vec, text_vec)

        assert output.shape == (1, 512)
        assert not output.requires_grad

    def test_batch_processing(self):
        """Test processing multiple batches"""
        model = PredictorNetwork()
        model.eval()

        num_batches = 5
        batch_size = 4

        with torch.no_grad():
            for _ in range(num_batches):
                vision_vec = torch.randn(batch_size, 512)
                text_vec = torch.randn(batch_size, 384)
                output = model(vision_vec, text_vec)

                assert output.shape == (batch_size, 512)


class TestPredictorNetworkSaveLoad:
    """Test suite for saving and loading models"""

    def test_state_dict_save_load(self, tmp_path):
        """Test that model can be saved and loaded"""
        # Create and save model
        model1 = PredictorNetwork()
        save_path = tmp_path / "predictor.pth"
        torch.save(model1.state_dict(), save_path)

        # Load model
        model2 = PredictorNetwork()
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # Set both models to eval mode (disables dropout for deterministic output)
        model1.eval()
        model2.eval()

        # Test that they produce same output
        vision_vec = torch.randn(2, 512)
        text_vec = torch.randn(2, 384)

        with torch.no_grad():
            output1 = model1(vision_vec, text_vec)
            output2 = model2(vision_vec, text_vec)

        assert torch.allclose(output1, output2)


# Benchmark test (optional, for performance measurement)
@pytest.mark.benchmark
class TestPredictorNetworkPerformance:
    """Performance benchmark tests"""

    def test_inference_speed(self):
        """Benchmark inference speed"""
        model = PredictorNetwork()
        model.eval()

        vision_vec = torch.randn(32, 512)
        text_vec = torch.randn(32, 384)

        import time
        start = time.time()

        with torch.no_grad():
            for _ in range(100):
                _ = model(vision_vec, text_vec)

        elapsed = time.time() - start
        avg_time = elapsed / 100

        print(f"\nAverage inference time (batch_size=32): {avg_time*1000:.2f}ms")
        assert avg_time < 0.1  # Should be less than 100ms per batch


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
