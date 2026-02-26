# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

**"Neuro-Search" - A Zero-Shot Video Search Engine**

Building a JEPA from scratch that maps video segments and text queries into a shared "meaning space."

## 🏆 Phase 1: COMPLETE! ✓

All four team members successfully built their neural network components:

| Team Member | Component | Output Dimension | Status |
|-------------|-----------|------------------|--------|
| **Nawfal** | Vision Module (ViT) | [Batch, 512] | ✓ Complete |
| **Ali** | Text Module (MiniLM) | [Batch, 384] | ✓ Complete |
| **Abdullah** | Predictor Network (4-layer MLP) | [Batch, 512] | ✓ Complete |
| **Ansab** | Loss & Training Pipeline | Cosine Similarity | ✓ Complete |

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Phase 1 Integration Test

**Demo Mode** (with dummy data):
```bash
python main.py --demo
```

**Interactive Mode** (with real images and your text):
```bash
python main.py --interactive
```

Place test images in `src/images/` for interactive mode!

## 📦 Project Structure

```
VL-JEPA/
├── main.py                      # Phase 1 integration test (THE MERGE FILE!)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── USAGE_GUIDE.md              # Detailed usage instructions
│
├── src/                         # All team modules
│   ├── __init__.py
│   ├── vision_module.py         # Nawfal's Vision Encoder
│   ├── TextModule.py            # Ali's Text Encoder
│   ├── predictor_network.py     # Abdullah's Predictor Network
│   ├── loss.py                  # Ansab's Loss Functions
│   ├── train.py                 # Ansab's Training Pipeline
│   └── images/                  # Test images for interactive mode
│       └── README.md            # Instructions for test images
│
└── tests/                       # Unit tests
    ├── test_loss.py
    └── test_predictor_network.py
```

## 🧩 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     VL-JEPA PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

Image (3x224x224)                Text ["cat playing"]
        ↓                                 ↓
   Vision Module                    Text Module
   (Nawfal)                         (Ali)
        ↓                                 ↓
    [Batch, 512]                     [Batch, 384]
        └──────────────┬──────────────────┘
                       ↓
              Concatenate [Batch, 896]
                       ↓
              Predictor Network (Abdullah)
              ┌────────────────────────┐
              │ Layer 1: 896 → 768     │
              │ Layer 2: 768 → 768     │
              │ Layer 3: 768 → 576     │
              │ Layer 4: 576 → 512     │
              └────────────────────────┘
                       ↓
              Prediction [Batch, 512]
                       ↓
              Cosine Similarity Loss (Ansab)
                       ↓
                   Backprop
```

## 🎯 The "Merge Day" Success Criteria

✅ **Phase 1 Complete:** All modules integrate without dimension mismatch errors!

```python
# main.py - The Integration File
img_vec = vision(image)              # Nawfal's module → [Batch, 512]
txt_vec = text_model(text)           # Ali's module → [Batch, 384]
pred = predictor(img_vec, txt_vec)   # Abdullah's module → [Batch, 512]
loss = train_step(...)               # Ansab's function → scalar loss

# ✓ NO DIMENSION MISMATCH ERRORS!
```

## 💡 Key Design Decisions

### Nawfal's Vision Module
- **Model**: Vision Transformer (ViT-Base) from `timm`
- **Frozen Weights**: No backpropagation (GPU-friendly)
- **Output**: 512-dimensional feature vectors

### Ali's Text Module
- **Model**: Sentence Transformer (MiniLM-L6-v2)
- **Built-in Tokenization**: Handles strings directly
- **Output**: 384-dimensional sentence embeddings

### Abdullah's Predictor Network
- **Architecture**: 4-layer MLP with bottleneck
- **Activation**: GELU (modern transformer standard)
- **Normalization**: LayerNorm
- **Regularization**: Dropout (0.1)

### Ansab's Training Pipeline
- **Loss Function**: Cosine Similarity Loss
- **Optimization**: Adam optimizer
- **Training Loop**: Clean gradient management

## 📊 Hardware Strategy

- **Coding**: Local laptops (VS Code)
- **Debugging**: Small tests locally (batch_size=2, 10 images)
- **Training**: Google Colab (Free T4 GPU)

## 🔧 Testing Individual Modules

Each module can be tested independently:

```bash
# Test Vision Module
python vision_module.py

# Test Text Module
python TextModule.py

# Test Predictor Network
python src/predictor_network.py

# Test Loss Functions
python tests/test_loss.py
```

## 📈 Next Steps: Phase 2

- [ ] Dataset Integration (real video data)
- [ ] Training on Google Colab
- [ ] Evaluation metrics
- [ ] Zero-shot search capabilities

## 🤝 Team Contributions

This project is a collaborative effort by the VL-JEPA team:
- **Nawfal**: Vision Encoder expertise
- **Ali**: NLP and Text Encoding
- **Abdullah**: Neural Network Architecture
- **Ansab**: Training Pipeline and Optimization

## 📚 References

- [JEPA Framework](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [Sentence Transformers](https://www.sbert.net/)
-m src.vision_module

# Test Text Module
python -m src.TextModule

# Test Predictor Network
python -m src.predictor_network

# Test Loss Functions
python -m pyteste**: Import errors
- **Solution**: Ensure `src/` is in Python path and all dependencies installed

---

**Status**: Phase 1 ✓ Complete | Ready for Phase 2 🚀
