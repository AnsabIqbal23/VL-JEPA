"""
Unit Tests for Loss Functions

Author: Ansab
Phase: 1 - Building Blocks Sprint

Test Coverage:
1. Cosine similarity loss basic functionality
2. Shape validation and error handling
3. Edge cases (identical vectors, orthogonal vectors)
4. Reduction modes (mean, sum, none)
5. JEPALoss class functionality
6. Gradient flow tests
7. Integration with training step
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loss import cosine_similarity_loss, CosineSimilarityLoss, JEPALoss


class TestCosineSimilarityLossFunction:
    """Test suite for the functional cosine_similarity_loss"""

    def test_basic_functionality(self):
        """Test basic loss computation"""
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss = cosine_similarity_loss(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert loss.item() <= 2  # Max loss is 2 (opposite vectors)

    def test_identical_vectors(self):
        """Test that identical vectors give loss ≈ 0"""
        vectors = torch.randn(4, 512)
        loss = cosine_similarity_loss(vectors, vectors)

        assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"

    def test_opposite_vectors(self):
        """Test that opposite vectors give loss ≈ 2"""
        vectors = torch.randn(4, 512)
        loss = cosine_similarity_loss(vectors, -vectors)

        assert abs(loss.item() - 2.0) < 1e-5, f"Expected ~2, got {loss.item()}"

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors give loss ≈ 1"""
        # Create orthogonal vectors using QR decomposition
        batch_size = 4
        dim = 512
        random_matrix = torch.randn(batch_size, dim * 2)
        q, _ = torch.linalg.qr(random_matrix.T)
        q = q.T

        vec1 = q[:, :dim]
        vec2 = q[:, dim:dim*2]

        loss = cosine_similarity_loss(vec1, vec2)
        # Orthogonal vectors should have cosine similarity ≈ 0, so loss ≈ 1
        assert abs(loss.item() - 1.0) < 0.1, f"Expected ~1, got {loss.item()}"

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError"""
        pred = torch.randn(4, 512)
        target = torch.randn(4, 256)  # Wrong dimension

        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity_loss(pred, target)

    def test_batch_size_mismatch_error(self):
        """Test that mismatched batch sizes raise ValueError"""
        pred = torch.randn(4, 512)
        target = torch.randn(8, 512)  # Wrong batch size

        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity_loss(pred, target)

    def test_reduction_modes(self):
        """Test different reduction modes"""
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)

        # Mean reduction (default)
        loss_mean = cosine_similarity_loss(pred, target, reduction='mean')
        assert loss_mean.dim() == 0

        # Sum reduction
        loss_sum = cosine_similarity_loss(pred, target, reduction='sum')
        assert loss_sum.dim() == 0

        # No reduction
        loss_none = cosine_similarity_loss(pred, target, reduction='none')
        assert loss_none.shape == (4,)

        # Mean of none should equal mean reduction
        assert torch.allclose(loss_none.mean(), loss_mean, atol=1e-6)

    def test_different_batch_sizes(self):
        """Test with various batch sizes"""
        for batch_size in [1, 2, 8, 16, 32]:
            pred = torch.randn(batch_size, 512)
            target = torch.randn(batch_size, 512)
            loss = cosine_similarity_loss(pred, target)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0

    def test_different_embedding_dims(self):
        """Test with various embedding dimensions"""
        for dim in [128, 256, 384, 512, 768, 1024]:
            pred = torch.randn(4, dim)
            target = torch.randn(4, dim)
            loss = cosine_similarity_loss(pred, target)

            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0


class TestCosineSimilarityLossClass:
    """Test suite for the CosineSimilarityLoss class"""

    @pytest.fixture
    def loss_fn(self):
        return CosineSimilarityLoss()

    def test_initialization(self):
        """Test class initialization with defaults"""
        loss_fn = CosineSimilarityLoss()
        assert loss_fn.reduction == 'mean'
        assert loss_fn.eps == 1e-8

    def test_custom_initialization(self):
        """Test class initialization with custom parameters"""
        loss_fn = CosineSimilarityLoss(reduction='sum', eps=1e-6)
        assert loss_fn.reduction == 'sum'
        assert loss_fn.eps == 1e-6

    def test_forward_pass(self, loss_fn):
        """Test forward pass"""
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_matches_functional(self, loss_fn):
        """Test that class output matches functional output"""
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)

        loss_class = loss_fn(pred, target)
        loss_func = cosine_similarity_loss(pred, target)

        assert torch.allclose(loss_class, loss_func, atol=1e-6)

    def test_invalid_reduction_mode(self):
        """Test that invalid reduction mode raises error"""
        loss_fn = CosineSimilarityLoss(reduction='invalid')
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)

        with pytest.raises(ValueError, match="Invalid reduction"):
            loss_fn(pred, target)


class TestJEPALoss:
    """Test suite for the JEPALoss class"""

    def test_cosine_mode(self):
        """Test JEPALoss in cosine mode"""
        loss_fn = JEPALoss(loss_type='cosine')
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_mse_mode(self):
        """Test JEPALoss in MSE mode"""
        loss_fn = JEPALoss(loss_type='mse')
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_combined_mode(self):
        """Test JEPALoss in combined mode"""
        loss_fn = JEPALoss(
            loss_type='combined',
            cosine_weight=0.8,
            mse_weight=0.2
        )
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error"""
        loss_fn = JEPALoss(loss_type='invalid')
        pred = torch.randn(4, 512)
        target = torch.randn(4, 512)

        with pytest.raises(ValueError, match="Unknown loss type"):
            loss_fn(pred, target)

    def test_extra_repr(self):
        """Test string representation"""
        loss_fn = JEPALoss(loss_type='combined', cosine_weight=0.7, mse_weight=0.3)
        repr_str = loss_fn.extra_repr()

        assert 'combined' in repr_str
        assert '0.7' in repr_str
        assert '0.3' in repr_str


class TestGradientFlow:
    """Test suite for gradient flow through loss functions"""

    def test_gradient_flow_functional(self):
        """Test gradients flow through functional loss"""
        pred = torch.randn(4, 512, requires_grad=True)
        target = torch.randn(4, 512)

        loss = cosine_similarity_loss(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_gradient_flow_class(self):
        """Test gradients flow through class-based loss"""
        loss_fn = CosineSimilarityLoss()
        pred = torch.randn(4, 512, requires_grad=True)
        target = torch.randn(4, 512)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_gradient_flow_jepa(self):
        """Test gradients flow through JEPALoss"""
        loss_fn = JEPALoss(loss_type='combined')
        pred = torch.randn(4, 512, requires_grad=True)
        target = torch.randn(4, 512)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_gradient_magnitude(self):
        """Test that gradients have reasonable magnitude"""
        pred = torch.randn(4, 512, requires_grad=True)
        target = torch.randn(4, 512)

        loss = cosine_similarity_loss(pred, target)
        loss.backward()

        # Gradients should not be too large or too small
        grad_norm = pred.grad.norm().item()
        assert 1e-6 < grad_norm < 1e3, f"Gradient norm {grad_norm} out of expected range"


class TestIntegration:
    """Integration tests with training components"""

    def test_with_simple_model(self):
        """Test loss with a simple neural network"""
        model = nn.Sequential(
            nn.Linear(896, 768),
            nn.GELU(),
            nn.Linear(768, 512)
        )

        vision_vec = torch.randn(4, 512)
        text_vec = torch.randn(4, 384)
        target = torch.randn(4, 512)

        # Forward pass
        combined = torch.cat([vision_vec, text_vec], dim=1)
        pred = model(combined)

        # Compute loss
        loss = cosine_similarity_loss(pred, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_training_loop_simulation(self):
        """Simulate a mini training loop"""
        model = nn.Linear(512, 512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = CosineSimilarityLoss()

        initial_loss = None
        for i in range(10):
            optimizer.zero_grad()

            pred = model(torch.randn(4, 512))
            target = torch.randn(4, 512)

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        # Just verify the loop completes without errors
        assert True


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with batch size of 1"""
        pred = torch.randn(1, 512)
        target = torch.randn(1, 512)
        loss = cosine_similarity_loss(pred, target)

        assert loss.dim() == 0

    def test_very_small_vectors(self):
        """Test with very small vector values"""
        pred = torch.randn(4, 512) * 1e-6
        target = torch.randn(4, 512) * 1e-6
        loss = cosine_similarity_loss(pred, target)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_very_large_vectors(self):
        """Test with very large vector values"""
        pred = torch.randn(4, 512) * 1e6
        target = torch.randn(4, 512) * 1e6
        loss = cosine_similarity_loss(pred, target)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_normalized_vectors(self):
        """Test with L2-normalized vectors"""
        pred = torch.randn(4, 512)
        pred = pred / pred.norm(dim=-1, keepdim=True)

        target = torch.randn(4, 512)
        target = target / target.norm(dim=-1, keepdim=True)

        loss = cosine_similarity_loss(pred, target)

        assert not torch.isnan(loss)
        assert 0 <= loss.item() <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
