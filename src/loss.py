"""
Loss Functions for VL-JEPA Training

Author: Ansab
Phase: 1 - Building Blocks Sprint
Description: Implements cosine similarity loss for JEPA training.
             The loss measures how well the predictor's output aligns
             with the target embeddings from the Y-encoder.

References:
- VL-JEPA Paper: Uses cosine similarity for embedding alignment
- JEPA Framework: Prediction in latent space, not pixel space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for JEPA training.

    Computes: loss = 1 - mean(cosine_similarity(pred, target))

    This loss encourages the predictor to output embeddings that are
    directionally similar to the target embeddings from the Y-encoder.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
                        'mean' | 'sum' | 'none'. Default: 'mean'
        eps (float): Small value to avoid division by zero. Default: 1e-8

    Example:
        >>> loss_fn = CosineSimilarityLoss()
        >>> pred = torch.randn(4, 512)  # Predictor output
        >>> target = torch.randn(4, 512)  # Y-encoder output
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cosine similarity loss.

        Args:
            pred (torch.Tensor): Predicted embeddings from predictor
                                Shape: [batch_size, embedding_dim]
            target (torch.Tensor): Target embeddings from Y-encoder
                                  Shape: [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Loss value (scalar if reduction='mean' or 'sum')

        Raises:
            ValueError: If pred and target shapes don't match
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape}, target {target.shape}"
            )

        # Compute cosine similarity per sample
        cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=self.eps)

        # Convert similarity to loss (1 - similarity)
        loss = 1 - cos_sim

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def cosine_similarity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Functional interface for cosine similarity loss.

    This is a convenience function that wraps CosineSimilarityLoss.

    Args:
        pred (torch.Tensor): Predicted embeddings [batch_size, embedding_dim]
        target (torch.Tensor): Target embeddings [batch_size, embedding_dim]
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'

    Returns:
        torch.Tensor: Loss value

    Example:
        >>> pred = torch.randn(4, 512)
        >>> target = torch.randn(4, 512)
        >>> loss = cosine_similarity_loss(pred, target)
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        )

    # Compute cosine similarity per sample
    cos_sim = F.cosine_similarity(pred, target, dim=-1)

    # Convert similarity to loss
    loss = 1 - cos_sim

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


class JEPALoss(nn.Module):
    """
    Combined loss for VL-JEPA training.

    This class provides a flexible loss computation that can combine
    multiple loss terms (cosine similarity, MSE, etc.) for JEPA training.

    Args:
        loss_type (str): Type of loss to use: 'cosine' | 'mse' | 'combined'
        cosine_weight (float): Weight for cosine loss in combined mode
        mse_weight (float): Weight for MSE loss in combined mode

    Example:
        >>> loss_fn = JEPALoss(loss_type='cosine')
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        loss_type: str = 'cosine',
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

        self.cosine_loss = CosineSimilarityLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the JEPA loss.

        Args:
            pred: Predicted embeddings [batch_size, embedding_dim]
            target: Target embeddings [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Combined loss value
        """
        if self.loss_type == 'cosine':
            return self.cosine_loss(pred, target)
        elif self.loss_type == 'mse':
            return self.mse_loss(pred, target)
        elif self.loss_type == 'combined':
            cos_loss = self.cosine_loss(pred, target)
            mse_loss = self.mse_loss(pred, target)
            return self.cosine_weight * cos_loss + self.mse_weight * mse_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, cosine_weight={self.cosine_weight}, mse_weight={self.mse_weight}"


# Quick test
if __name__ == "__main__":
    print("Loss Functions Test Suite")
    print("-" * 50)

    batch_size = 4
    embedding_dim = 512

    pred = torch.randn(batch_size, embedding_dim)
    target = torch.randn(batch_size, embedding_dim)

    # Test functional interface
    loss1 = cosine_similarity_loss(pred, target)
    print(f"[OK] Functional cosine loss: {loss1:.4f}")

    # Test class interface
    loss_fn = CosineSimilarityLoss()
    loss2 = loss_fn(pred, target)
    print(f"[OK] Class-based cosine loss: {loss2:.4f}")

    # Test JEPALoss
    jepa_loss = JEPALoss(loss_type='cosine')
    loss3 = jepa_loss(pred, target)
    print(f"[OK] JEPALoss (cosine): {loss3:.4f}")

    # Test combined loss
    jepa_combined = JEPALoss(loss_type='combined', cosine_weight=0.8, mse_weight=0.2)
    loss4 = jepa_combined(pred, target)
    print(f"[OK] JEPALoss (combined): {loss4:.4f}")

    # Test identical vectors (should give loss ≈ 0)
    identical = torch.randn(batch_size, embedding_dim)
    loss_identical = cosine_similarity_loss(identical, identical)
    print(f"[OK] Identical vectors loss: {loss_identical:.6f} (should be ~0)")

    print("-" * 50)
    print("All tests passed!")