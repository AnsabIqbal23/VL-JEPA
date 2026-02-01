# Ansab's Work: Loss Functions & Training Pipeline - Phase 1

## Overview

This branch contains Ansab's implementation of the **Loss Functions** and **Training Pipeline** for the VL-JEPA project. These components are essential for training the model to predict target embeddings from vision-language inputs.

## Task Summary (Phase 1)

**Role**: The "Training Brain"
**Goal**: Implement loss functions and training utilities that orchestrate the learning process
**Skills Gained**: Loss function design, PyTorch training loops, gradient management

### What These Modules Do

```
Training Pipeline Flow:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Image ──→ [Vision Encoder] ──→ Vision Embedding (512-dim)
                                        │
                                        ├──→ [Predictor] ──→ Predicted (512-dim)
                                        │         ↓
Text ───→ [Text Encoder] ───→ Text Embedding (384-dim)
                                                  │
                                                  ↓
Target Embedding (512-dim) ←─── [Y-Encoder] ←── Target
                                                  │
                                                  ↓
                              Loss = 1 - CosineSimilarity(Predicted, Target)
                                                  │
                                                  ↓
                                            Backpropagate
                                                  │
                                                  ↓
                                          Update Weights
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Architecture Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Primary Loss** | Cosine Similarity | Measures directional alignment in embedding space; scale-invariant |
| **Loss Formula** | `1 - cos_sim` | Converts similarity (higher=better) to loss (lower=better) |
| **Reduction** | Mean | Averages across batch for stable gradients |
| **Alternative** | MSE Loss | Available for experiments; measures absolute distance |
| **Combined Mode** | Weighted sum | Allows mixing cosine + MSE for ablation studies |

## Project Structure

```
VL-JEPA/
├── src/
│   ├── __init__.py           # Package exports
│   ├── loss.py               # Loss functions (CosineSimilarityLoss, JEPALoss)
│   └── train.py              # Training utilities (train_step, validate_step)
├── tests/
│   └── test_loss.py          # Unit tests (40+ tests)
├── main.py                   # Integration test with dummy modules
├── requirements.txt          # Dependencies
└── README_Ansab.md           # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Test loss functions
python src/loss.py

# Run integration test
python main.py

# Run full test suite
pytest tests/test_loss.py -v
```

### 3. Basic Usage Examples

#### Using the Loss Function

```python
import torch
from src.loss import cosine_similarity_loss, CosineSimilarityLoss, JEPALoss

# Functional interface
pred = torch.randn(4, 512)    # Predictor output
target = torch.randn(4, 512)  # Y-encoder output
loss = cosine_similarity_loss(pred, target)
print(f"Loss: {loss.item():.4f}")

# Class-based interface
loss_fn = CosineSimilarityLoss()
loss = loss_fn(pred, target)

# JEPALoss with different modes
loss_fn = JEPALoss(loss_type='cosine')      # Default
loss_fn = JEPALoss(loss_type='mse')         # MSE only
loss_fn = JEPALoss(loss_type='combined',    # Weighted combination
                   cosine_weight=0.8,
                   mse_weight=0.2)
```

#### Using the Training Step

```python
from src.train import train_step, validate_step

# Single training step
loss = train_step(
    image=batch_images,           # [B, 3, 224, 224]
    text=batch_texts,             # List of strings
    target=target_embeddings,     # [B, 512]
    vision_model=vision_encoder,  # Nawfal's module
    text_model=text_encoder,      # Ali's module
    predictor=predictor_network,  # Abdullah's module
    optimizer=optimizer
)

# Validation step (no gradients)
metrics = validate_step(
    image, text, target,
    vision_model, text_model, predictor
)
print(f"Val Loss: {metrics['loss']:.4f}")
print(f"Cosine Sim: {metrics['cosine_similarity']:.4f}")
```

## Testing Coverage

The test suite includes **40+ tests** covering:

### Loss Function Tests
- Basic functionality
- Identical vectors (loss ≈ 0)
- Opposite vectors (loss ≈ 2)
- Orthogonal vectors (loss ≈ 1)
- Shape validation and error handling
- Different reduction modes (mean, sum, none)
- Various batch sizes and embedding dimensions

### Gradient Flow Tests
- Gradients flow correctly
- No NaN gradients
- Reasonable gradient magnitudes

### Integration Tests
- With simple neural network
- Training loop simulation

### Edge Cases
- Single sample batch
- Very small/large vector values
- L2-normalized vectors

### Run Specific Test Categories

```bash
# Run only functional loss tests
pytest tests/test_loss.py::TestCosineSimilarityLossFunction -v

# Run only gradient tests
pytest tests/test_loss.py::TestGradientFlow -v

# Run only JEPALoss tests
pytest tests/test_loss.py::TestJEPALoss -v
```

## API Reference

### `cosine_similarity_loss(pred, target, reduction='mean')`

Functional interface for cosine similarity loss.

**Parameters:**
- `pred` (Tensor): Predicted embeddings `[batch_size, embedding_dim]`
- `target` (Tensor): Target embeddings `[batch_size, embedding_dim]`
- `reduction` (str): `'mean'` | `'sum'` | `'none'`

**Returns:** Loss tensor

### `CosineSimilarityLoss(reduction='mean', eps=1e-8)`

Class-based cosine similarity loss.

**Parameters:**
- `reduction` (str): Reduction mode
- `eps` (float): Epsilon for numerical stability

### `JEPALoss(loss_type='cosine', cosine_weight=1.0, mse_weight=0.0)`

Flexible loss for JEPA training.

**Parameters:**
- `loss_type` (str): `'cosine'` | `'mse'` | `'combined'`
- `cosine_weight` (float): Weight for cosine loss in combined mode
- `mse_weight` (float): Weight for MSE loss in combined mode

### `train_step(...)`

Execute single training step.

**Parameters:**
- `image` (Tensor): Batch of images
- `text` (List[str]): Batch of text queries
- `target` (Tensor): Target embeddings
- `vision_model` (Module): Vision encoder
- `text_model` (Module): Text encoder
- `predictor` (Module): Predictor network
- `optimizer` (Optimizer): Optimizer
- `loss_fn` (Module, optional): Custom loss function

**Returns:** Loss value (float)

### `validate_step(...)`

Execute single validation step (no gradients).

**Returns:** Dict with `'loss'` and `'cosine_similarity'`

## Integration with Team Components

### With Abdullah's Predictor

```python
from src.predictor_network import PredictorNetwork
from src.loss import JEPALoss
from src.train import train_step

predictor = PredictorNetwork(
    vision_dim=512,   # Nawfal's output
    text_dim=384,     # Ali's output
    output_dim=512    # Target space
)

loss_fn = JEPALoss(loss_type='cosine')
optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    loss = train_step(
        batch['image'], batch['text'], batch['target'],
        vision_model, text_model, predictor,
        optimizer, loss_fn
    )
```

### Dimension Compatibility

| Component | Output Dim | Status |
|-----------|------------|--------|
| Nawfal's Vision Encoder | 512 | ✅ Verified |
| Ali's Text Encoder | 384 | ✅ Verified |
| Abdullah's Predictor | 512 | ✅ Verified |
| Target (Y-Encoder) | 512 | ✅ Matches |

## Phase 2 Preview: Experiments

### Loss Function Ablation

| Config | Loss Type | Expected Use Case |
|--------|-----------|-------------------|
| A | Cosine only | Direction alignment (default) |
| B | MSE only | Magnitude matching |
| C | Combined (0.8/0.2) | Balanced approach |
| D | Combined (0.5/0.5) | Equal weighting |

### Metrics to Track

- Training Loss (per epoch)
- Validation Loss (per epoch)
- Cosine Similarity (should increase during training)
- Gradient Norms (for stability monitoring)

## Known Issues & Limitations

1. **No Learning Rate Scheduler**: Will be added in Phase 2
2. **No Gradient Clipping**: May need for training stability
3. **No Mixed Precision**: Could add for faster training on GPU
4. **Single GPU Only**: No distributed training support yet

## Learning Resources

Based on this implementation, I learned:

1. **Loss Function Design**:
   - Cosine similarity for embedding alignment
   - Converting similarity to loss
   - Reduction modes and their effects

2. **PyTorch Training Patterns**:
   - Zero gradients before forward pass
   - Proper backward pass
   - Optimizer step ordering

3. **Testing Neural Network Components**:
   - Gradient flow verification
   - Edge case handling
   - Numerical stability tests

## CV Bullet Points

> **Neuro-Search: Semantic Video Understanding Engine**
> *PyTorch, Loss Functions, Training Pipelines*
>
> - Implemented cosine similarity loss function for Joint Embedding Predictive Architecture (JEPA), enabling directional alignment of multimodal embeddings
> - Developed flexible JEPALoss class supporting cosine, MSE, and weighted combination modes for training ablation studies
> - Created comprehensive training pipeline with train_step and validate_step functions, properly handling gradient computation and optimizer updates
> - Built robust test suite (40+ tests) covering loss computation, gradient flow, edge cases, and integration scenarios

## Contact & Collaboration

**Developer**: Ansab
**Branch**: `development/Ansab`
**Phase**: 1 - Building Blocks Sprint
**Status**: ✅ Complete (Ready for Phase 1 Merge)

---

**Next Steps**:
1. Merge with team members' modules
2. Run integration tests with real encoders
3. Phase 2: Add learning rate schedulers and gradient clipping
4. Phase 3: Implement full training loop with checkpointing

---

*Last Updated: January 26, 2025*
