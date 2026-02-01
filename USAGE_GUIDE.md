# VL-JEPA Usage Guide

## Quick Start - Running the Complete Pipeline

We've created two main scripts to run the complete VL-JEPA pipeline:

### 1. Quick Demo (Recommended for First Test)

Run the complete pipeline with dummy data - no image files needed:

```bash
python quick_demo.py
```

This will:
- ✓ Initialize all modules (Vision, Text, Predictor, Loss)
- ✓ Process sample image-text pairs
- ✓ Show embeddings and similarity scores
- ✓ Display the complete pipeline flow

### 2. Full Inference Script

For real image-text matching:

**Option A: Demo Mode** (no image file needed)
```bash
python inference.py --demo
```

**Option B: With Real Images**
```bash
python inference.py --image path/to/your/image.jpg --text "a cat sitting"
```

## What Each File Does

| File | Purpose | Who Built It |
|------|---------|--------------|
| `quick_demo.py` | Complete pipeline demo with dummy data | Integration |
| `inference.py` | Full inference with real images | Integration |
| `main.py` | Training pipeline test | Ansab |
| `TextModule.py` | Text encoder (384-dim) | Ali |
| `vision_module.py` | Vision encoder (512-dim) | Nawfal |
| `src/predictor_network.py` | Predictor network (4-layer MLP) | Abdullah |
| `src/loss.py` | Cosine similarity loss | Ansab |
| `src/train.py` | Training utilities | Ansab |

## Pipeline Flow

```
User Input:
  ├─ Image (file or tensor) → Vision Module → [Batch, 512] embedding
  └─ Text (string) → Text Module → [Batch, 384] embedding
                                        ↓
                           Concatenate [512 + 384 = 896]
                                        ↓
                           Predictor Network (4 layers)
                                        ↓
                           Joint Embedding [Batch, 512]
                                        ↓
                    Compare with Target → Similarity Score
```

## Installation

Make sure you have all dependencies:

```bash
pip install torch torchvision transformers timm pillow
```

## Examples

### Example 1: Test the integration
```bash
python quick_demo.py
```

### Example 2: Demo mode
```bash
python inference.py --demo
```

### Example 3: Real image inference
```bash
python inference.py --image cat.jpg --text "a fluffy cat"
```

### Example 4: Run training test
```bash
python main.py
```

## Output Example

When you run `quick_demo.py`, you'll see:

```
================================================================================
                       VL-JEPA COMPLETE PIPELINE DEMO                        
================================================================================

Using device: cuda

--------------------------------------------------------------------------------
STEP 1: Initializing Modules
--------------------------------------------------------------------------------

[1/4] Vision Module (Nawfal's module)...
      ✓ Vision Module ready - outputs [Batch, 512]

[2/4] Text Module (Ali's module)...
      ✓ Text Module ready - outputs [Batch, 384]

[3/4] Predictor Network (Abdullah's module)...
      ✓ Predictor Network ready - outputs [Batch, 512]

[4/4] Loss Function (Ansab's module)...
      ✓ Cosine Similarity Loss ready

... (processing steps)

✓ Average Similarity: 75.32%
✓ Loss: 0.2468

DEMO COMPLETED SUCCESSFULLY!
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd d:\git\VL-JEPA
python quick_demo.py
```

### Module Not Found
The scripts automatically add paths, but if you have issues:
```python
import sys
sys.path.insert(0, 'path/to/VL-JEPA')
sys.path.insert(0, 'path/to/VL-JEPA/src')
```

### CUDA Out of Memory
If you get CUDA errors, the scripts will automatically fall back to CPU, or you can force CPU:
```python
device = 'cpu'  # in the script
```

## Next Steps

1. ✅ Test with `quick_demo.py` to verify all modules work
2. ✅ Test with `inference.py --demo` for interactive demo
3. Try with your own images using `inference.py --image your_image.jpg --text "description"`
4. Check `main.py` to see how training works
5. Start training with real data (Phase 2)

## Module Details

### Vision Module
- Input: RGB images [Batch, 3, 224, 224]
- Model: ViT-Base (Vision Transformer)
- Output: [Batch, 512] embeddings
- Frozen: ✓ (no training)

### Text Module  
- Input: List of text strings
- Model: MiniLM-L6-v2 (Sentence Transformer)
- Output: [Batch, 384] embeddings
- Frozen: ✓ (no training)

### Predictor Network
- Input: Concatenated [vision, text] = [Batch, 896]
- Architecture: 4-layer MLP (896 → 768 → 768 → 576 → 512)
- Output: [Batch, 512] joint embeddings
- Trainable: ✓ (this is what we train!)

### Loss Function
- Type: Cosine Similarity Loss
- Formula: loss = 1 - mean(cosine_similarity(pred, target))
- Goal: Minimize loss = maximize similarity
