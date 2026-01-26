# Abdullah's Work: PredictorNetwork - Phase 1

## 🎯 Overview

This branch contains Abdullah's implementation of the **PredictorNetwork**, the core JEPA logic for the VL-JEPA project. The network learns to predict target embeddings from concatenated vision and text vectors.

## 📋 Task Summary (Phase 1)

**Role**: The "Predictor Brain"
**Goal**: Build a 4-layer Neural Network that processes multimodal inputs
**Skills Gained**: Designing Neural Networks from scratch in PyTorch

### What This Module Does

```
Input: [Vision Vector (512-dim), Text Vector (384-dim)]
         ↓ Concatenation
    Combined Vector (896-dim)
         ↓ Layer 1: Linear + LayerNorm + GELU + Dropout
    Hidden (768-dim)
         ↓ Layer 2: Linear + LayerNorm + GELU + Dropout
    Hidden (768-dim)
         ↓ Layer 3: Linear + LayerNorm + GELU + Dropout
    Bottleneck (576-dim)  [configurable via bottleneck_ratio=0.75]
         ↓ Layer 4: Linear
Output: Predicted Embedding (512-dim)  [optionally L2-normalized]
```

## 🏗️ Architecture Design Decisions

Based on analysis of the VL-JEPA research paper, I made these choices:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Activation** | GELU | Used in modern transformers (BERT, Llama); smoother than ReLU |
| **Normalization** | LayerNorm | Stable for transformer-like architectures; better than BatchNorm for variable batch sizes |
| **Regularization** | Dropout (0.1) | Prevents overfitting; 0.1 is standard for transformers; applied to all layers |
| **Hidden Dims** | 768 → 768 → 576 | Matches BERT-base; gentle bottleneck (75%) preserves information better |
| **Initialization** | Xavier Uniform | Helps gradient flow in deep networks |
| **Output Norm** | Optional L2 | Can normalize embeddings for contrastive learning |

## 📁 Project Structure

```
VL-JEPA/
├── src/
│   └── predictor_network.py      # Main implementation (400+ lines)
├── tests/
│   └── test_predictor_network.py # Unit tests (450+ lines, 28 tests)
├── requirements.txt              # Dependencies
├── README_Abdullah.md            # This file
└── .gitignore
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Tests

```bash
# Test the network with dummy data
python src/predictor_network.py

# Run full test suite
pytest tests/test_predictor_network.py -v

# Run with coverage report
pytest tests/test_predictor_network.py --cov=src --cov-report=html
```

### 3. Basic Usage Example

```python
import torch
from src.predictor_network import PredictorNetwork

# Initialize the network
model = PredictorNetwork(
    vision_dim=512,   # From Nawfal's VisionModule
    text_dim=384,     # From Ali's TextModule
    hidden_dim=768,
    output_dim=512
)

# Print architecture summary
model.print_architecture()

# Create dummy input data
batch_size = 4
vision_vec = torch.randn(batch_size, 512)  # From Nawfal
text_vec = torch.randn(batch_size, 384)    # From Ali

# Forward pass
predicted_embedding = model(vision_vec, text_vec)
print(f"Output shape: {predicted_embedding.shape}")  # [4, 512]

# With intermediate outputs (for debugging/visualization)
output, intermediates = model(vision_vec, text_vec, return_intermediate=True)
print(f"Intermediate layers: {list(intermediates.keys())}")
```

## 🧪 Testing Coverage

The test suite includes **28 tests** covering:

✅ **Initialization Tests**
- Default parameter initialization
- Custom dimension configurations
- Residual connection setup
- Normalize output parameter
- Bottleneck ratio parameter

✅ **New Features Tests**
- Output L2 normalization verification
- Non-normalized output verification
- Custom bottleneck ratio dimensions
- Bottleneck ratio parameter count impact

✅ **Forward Pass Tests**
- Output shape validation
- Different batch sizes (1, 2, 8, 16, 32)
- Intermediate activation outputs
- NaN detection

✅ **Error Handling Tests**
- Wrong input dimensions (should raise ValueError)
- Batch size mismatches
- Invalid configurations

✅ **Gradient Tests**
- Gradient flow verification
- Backward pass validation
- Eval mode (no gradient computation)

✅ **Integration Tests**
- End-to-end pipeline (forward → loss → backward)
- Save/load model state
- Batch processing
- Inference mode

✅ **Performance Tests**
- Inference speed benchmarks
- Parameter counting
- Memory usage

### Run Specific Test Categories

```bash
# Run only initialization tests
pytest tests/test_predictor_network.py::TestPredictorNetworkInitialization -v

# Run only forward pass tests
pytest tests/test_predictor_network.py::TestPredictorNetworkForward -v

# Run performance benchmarks
pytest tests/test_predictor_network.py -m benchmark -v
```

## 📊 Model Specifications

### Default Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vision Input Dim | 512 | From Nawfal's VisionModule |
| Text Input Dim | 384 | From Ali's TextModule |
| Hidden Dim | 768 | BERT-base size |
| Bottleneck Ratio | 0.75 | Gentle compression (768 → 576) |
| Output Dim | 512 | Target embedding space |
| Dropout Rate | 0.1 | Standard for transformers (all layers) |
| Normalize Output | False | Optional L2 normalization |
| Total Parameters | ~2.0M | Trainable parameters |

### Pre-configured Variants (for Phase 2)

```python
from src.predictor_network import PredictorNetworkConfig

# Default config (bottleneck_ratio=0.75)
config = PredictorNetworkConfig.get_config('DEFAULT')
model = PredictorNetwork(**config)

# Wide network (for Ali's Phase 2 experiment)
config = PredictorNetworkConfig.get_config('WIDE')
model = PredictorNetwork(**config)  # hidden_dim=1024

# Deep network with residual and normalization (for contrastive learning)
config = PredictorNetworkConfig.get_config('DEEP')
model = PredictorNetwork(**config)  # use_residual=True, normalize_output=True, bottleneck_ratio=0.83
```

### New Parameters (v2.0)

```python
# Enable L2 normalization for contrastive learning
model = PredictorNetwork(normalize_output=True)

# Customize bottleneck compression
model = PredictorNetwork(bottleneck_ratio=0.5)  # More aggressive (768 → 384)
model = PredictorNetwork(bottleneck_ratio=0.9)  # Gentler (768 → 691)
```

## 🔧 Integration with Team Components

### With Nawfal's VisionModule

```python
# Nawfal's code (expected interface)
from vision_module import VisionModule

vision_encoder = VisionModule()
image_tensor = load_image("video_frame.jpg")
vision_vec = vision_encoder(image_tensor)  # Output: [batch, 512]

# My predictor
from predictor_network import PredictorNetwork
predictor = PredictorNetwork()
```

### With Ali's TextModule

```python
# Ali's code (expected interface)
from text_module import TextModule

text_encoder = TextModule()
query_text = "What is happening in this video?"
text_vec = text_encoder(query_text)  # Output: [batch, 384]

# My predictor
predicted_embedding = predictor(vision_vec, text_vec)
```

### With Ansab's Loss Function

```python
# Ansab's code (expected interface)
from loss_function import JEPALoss

loss_fn = JEPALoss()

# Forward pass
predicted = predictor(vision_vec, text_vec)
target = y_encoder(target_text)

# Calculate loss
loss = loss_fn(predicted, target)
loss.backward()
```

## 📈 Phase 2 Preview: Hyperparameter Experiments

In Phase 2, I will run experiments comparing:

### Optimizer Variants

| Config | Optimizer | Learning Rate | Y-Encoder LR | Expected Outcome |
|--------|-----------|---------------|--------------|------------------|
| A | AdamW | 1e-3 | 5e-5 | Fast convergence, might oscillate |
| B | AdamW | 1e-4 | 5e-6 | Stable, slower convergence |
| C | SGD+Momentum | 1e-3 | 5e-5 | Sharp minima, good generalization |
| D | SGD+Momentum | 1e-4 | 5e-6 | Very stable, very slow |

### Tracking Metrics

For each experiment, I'll track:
- Training Loss (per epoch)
- Validation Loss (per epoch)
- Top-5 Accuracy (video classification)
- CIDEr Score (video captioning)
- Training Time per Epoch
- GPU Memory Usage

## 🐛 Known Issues & Limitations

1. **Hardware Constraints**:
   - Current implementation runs on CPU (no GPU on laptops)
   - For training, must use Google Colab with T4 GPU

2. **Simplified Architecture**:
   - Research paper uses 8 Transformer layers (~490M params)
   - Our implementation uses 4 MLP layers (~2.0M params)
   - This is intentional for educational purposes and hardware constraints

3. **Input Dimension Assumptions**:
   - Default expects vision_dim=512, text_dim=384 (configurable)
   - Validated: Nawfal's output = 512-dim, Ali's output = 384-dim

## 📚 Learning Resources

Based on this implementation, I learned:

1. **PyTorch Module Design**:
   - Custom `nn.Module` creation
   - Weight initialization strategies
   - Forward pass implementation

2. **Deep Learning Best Practices**:
   - LayerNorm vs BatchNorm trade-offs
   - GELU vs ReLU activation functions
   - Dropout placement
   - Residual connections

3. **Testing & Validation**:
   - Pytest framework
   - Test fixtures and parameterization
   - Gradient flow verification
   - Shape validation

4. **Research Paper Implementation**:
   - Translating paper architecture to code
   - Making simplifications for constraints
   - Documenting design decisions

## 🎓 CV Bullet Points

> **Neuro-Search: Semantic Video Understanding Engine**
> *PyTorch, Transformers, Neural Network Design, Testing*
>
> - Architected a 4-layer MLP predictor network (~2.0M parameters) for Joint Embedding Predictive Architecture (JEPA), processing multimodal vision-language inputs
> - Implemented modular PyTorch architecture with LayerNorm, GELU activation, configurable bottleneck ratio, optional L2 normalization, and residual connections
> - Developed comprehensive test suite (28 unit tests) covering initialization, forward pass, error handling, gradient flow, new features, and integration scenarios
> - Designed three architectural variants (DEFAULT, WIDE, DEEP) with configurable bottleneck compression to support Phase 2 ablation studies

## 🤝 Team Coordination

### Questions for Team (Phase 1 Merge Day)

1. **For Nawfal**: ✅ RESOLVED
   - Output shape: `[batch_size, 512]` - matches our `vision_dim=512`

2. **For Ali**: ✅ RESOLVED
   - Output shape: `[batch_size, 384]` - matches our `text_dim=384`

3. **For Ansab**:
   - Should I normalize my output embeddings (L2 norm)? → Added `normalize_output` option
   - What loss function are you implementing (Cosine? InfoNCE?)?

### Integration Checklist

Before "Merge Day", we need to verify:

- [x] Nawfal's vision output shape matches my `vision_dim` parameter (512)
- [x] Ali's text output shape matches my `text_dim` parameter (384)
- [ ] Ansab's loss function accepts my output tensor format
- [ ] All modules use the same batch dimension convention
- [ ] We agree on tensor device (CPU vs CUDA) handling

## 📞 Contact & Collaboration

**Developer**: Abdullah
**Branch**: `development/Abdullah`
**Phase**: 1 - Building Blocks Sprint
**Status**: ✅ Complete (Ready for Phase 1 Merge)

---

**Next Steps**:
1. Wait for team members to complete their modules
2. Merge Day: Test integration in `main.py`
3. Phase 2: Run optimizer experiments on Google Colab
4. Phase 3: Build Streamlit frontend dashboard

---

*Last Updated: January 26, 2025 (v2.0 - Added bottleneck_ratio, normalize_output, fixed layer 3 dropout)*
