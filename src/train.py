"""
Training Utilities for VL-JEPA

Author: Ansab
Phase: 1 - Building Blocks Sprint
Description: Provides training step functions for the VL-JEPA model.
             Orchestrates the forward pass through all encoders and predictor,
             computes loss, and updates model parameters.

Training Flow:
    1. Image → Vision Encoder → Vision Embedding (512-dim)
    2. Text → Text Encoder → Text Embedding (384-dim)
    3. [Vision, Text] → Predictor → Predicted Embedding (512-dim)
    4. Loss = CosineSimilarity(Predicted, Target)
    5. Backpropagate and update parameters
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict, Any
from loss import cosine_similarity_loss, JEPALoss


def train_step(
    image: torch.Tensor,
    text: Union[List[str], torch.Tensor],
    target: torch.Tensor,
    vision_model: nn.Module,
    text_model: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None
) -> float:
    """
    Execute a single training step for VL-JEPA.

    This function orchestrates one complete forward-backward pass:
    1. Encodes image through vision model
    2. Encodes text through text model
    3. Predicts target embedding through predictor
    4. Computes loss between prediction and target
    5. Backpropagates gradients and updates parameters

    Args:
        image (torch.Tensor): Batch of images [batch_size, 3, H, W]
        text (List[str] | torch.Tensor): Batch of text queries
        target (torch.Tensor): Target embeddings from Y-encoder [batch_size, output_dim]
        vision_model (nn.Module): Vision encoder (Nawfal's module)
        text_model (nn.Module): Text encoder (Ali's module)
        predictor (nn.Module): Predictor network (Abdullah's module)
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        loss_fn (nn.Module, optional): Loss function. Defaults to cosine_similarity_loss

    Returns:
        float: Loss value for this training step

    Example:
        >>> loss = train_step(
        ...     image=batch_images,
        ...     text=batch_texts,
        ...     target=target_embeddings,
        ...     vision_model=vision_encoder,
        ...     text_model=text_encoder,
        ...     predictor=predictor_network,
        ...     optimizer=optimizer
        ... )
    """
    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass through encoders
    img_vector = vision_model(image)      # [batch_size, vision_dim]
    text_vector = text_model(text)        # [batch_size, text_dim]

    # Forward pass through predictor
    pred = predictor(img_vector, text_vector)  # [batch_size, output_dim]

    # Compute loss
    if loss_fn is not None:
        loss = loss_fn(pred, target)
    else:
        loss = cosine_similarity_loss(pred, target)

    # Backpropagate gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    return loss.item()


def validate_step(
    image: torch.Tensor,
    text: Union[List[str], torch.Tensor],
    target: torch.Tensor,
    vision_model: nn.Module,
    text_model: nn.Module,
    predictor: nn.Module,
    loss_fn: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Execute a single validation step (no gradient computation).

    Args:
        image: Batch of images [batch_size, 3, H, W]
        text: Batch of text queries
        target: Target embeddings [batch_size, output_dim]
        vision_model: Vision encoder
        text_model: Text encoder
        predictor: Predictor network
        loss_fn: Loss function (optional)

    Returns:
        Dict containing 'loss' and 'cosine_similarity' metrics
    """
    with torch.no_grad():
        # Forward pass
        img_vector = vision_model(image)
        text_vector = text_model(text)
        pred = predictor(img_vector, text_vector)

        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(pred, target)
        else:
            loss = cosine_similarity_loss(pred, target)

        # Compute cosine similarity (for metrics)
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()

    return {
        'loss': loss.item(),
        'cosine_similarity': cos_sim.item()
    }


def train_epoch(
    dataloader: Any,
    vision_model: nn.Module,
    text_model: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Train for one complete epoch.

    Args:
        dataloader: DataLoader yielding (image, text, target) batches
        vision_model: Vision encoder
        text_model: Text encoder
        predictor: Predictor network
        optimizer: Optimizer
        loss_fn: Loss function (optional)
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Dict containing 'avg_loss' for the epoch
    """
    predictor.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        image, text, target = batch
        image = image.to(device)
        target = target.to(device)

        loss = train_step(
            image, text, target,
            vision_model, text_model, predictor,
            optimizer, loss_fn
        )
        total_loss += loss
        num_batches += 1

    return {'avg_loss': total_loss / max(num_batches, 1)}