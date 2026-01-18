import torch
import torch.nn.functional as F

def cosine_similarity_loss(pred, target):

    assert pred.shape == target.shape, (
        f"Shape Mismatch: pred {pred.shape}, target {target.shape}."
    )

    # For finding cosine similarity per sample
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    
    # Convert similarity to loss
    loss = 1 - cos_sim.mean()
    return loss