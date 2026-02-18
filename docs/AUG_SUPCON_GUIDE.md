# Augmented Supervised Contrastive Learning (Aug-SupCon)

## Overview

This module implements **Augmented Supervised Contrastive Learning** for the COLLIDE2V dataset, combining:
1. **Data Augmentation**: Random masking to create multiple views of each sample
2. **SimCLR Loss**: Contrastive learning that treats augmented views as positives
3. **Supervised Learning**: Uses class labels to define positive/negative pairs

## Key Differences from Vanilla SupCon

| Feature | Vanilla SupCon | Aug-SupCon |
|---------|---------------|------------|
| **Augmentation** | None | Random masking (feature or particle-level) |
| **Loss Function** | SupConLoss | SimCLRLoss |
| **Positive Pairs** | Same-class samples only | Same-class + augmented views |
| **Robustness** | Lower | Higher (trained on perturbed data) |
| **Training Time** | Faster | Slower (2x forward passes) |

## How It Works

### 1. Augmentation Strategy

Two masking modes available:

#### **Feature-Level Masking** (`mask_full_particle=False`)
- Randomly masks individual features with probability `p`
- More aggressive augmentation
- Example: Randomly zero out pT, η, φ values independently

```python
# Input:  [jet1_pt, jet1_eta, jet1_phi, jet2_pt, jet2_eta, ...]
# Output: [jet1_pt, 0.0,      jet1_phi, 0.0,     jet2_eta, ...]
#                    ↑ masked           ↑ masked
```

#### **Particle-Level Masking** (`mask_full_particle=True`)
- Randomly masks entire particles (all features of an object)
- Simulates detector inefficiencies or missing particles
- Example: Completely remove jet2 from the event

```python
# Input:  [jet1_pt, jet1_eta, jet1_phi, jet2_pt, jet2_eta, jet2_phi, ...]
# Output: [jet1_pt, jet1_eta, jet1_phi, 0.0,     0.0,      0.0,      ...]
#                                        ↑ all jet2 features masked
```

### 2. Training Process

For each batch:

1. **Create Augmented Views**
   ```
   Original: x → [x1, x2, ..., xN]
   View 1:   Augment(x) → [x1', x2', ..., xN']
   View 2:   Augment(x) → [x1'', x2'', ..., xN'']
   ```

2. **Forward Pass**
   ```
   Concatenate views → [x1', x1'', x2', x2'', ..., xN', xN'']
   ↓
   Encoder → embeddings
   ↓
   Projection Head → projections (L2-normalized)
   ```

3. **Compute Loss**
   - For sample i with label y_i:
     - **Positives**: All samples j where y_j == y_i (including augmented views)
     - **Negatives**: All samples j where y_j != y_i
   - Pull positives together, push negatives apart

### 3. SimCLR Loss Formula

```
Loss = -τ * mean_over_samples[ 
    mean_over_positives[ 
        log(exp(sim(i,p)) / sum_all_k(exp(sim(i,k)))) 
    ]
]
```

Where:
- `sim(i,j)` = cosine similarity between L2-normalized embeddings
- `τ` = temperature (controls hardness of negative mining)
- Positives include both same-class samples AND augmented views

## Usage

### Quick Start

```bash
# Train with default configuration
python src/train.py experiment=aug_supcon_6class

# Override specific parameters
python src/train.py experiment=aug_supcon_6class \
    model.mask_probability=0.3 \
    model.temperature=0.05 \
    model.mask_full_particle=true
```

### Configuration Files

1. **Model Config**: `configs/model/augmented_supcon.yaml`
   - Architecture parameters
   - Augmentation settings
   - Loss hyperparameters

2. **Experiment Config**: `configs/experiment/aug_supcon_6class.yaml`
   - Full experiment setup
   - Data configuration
   - Training settings

### Python API

```python
from src.models.collide2v_augmented_supcon import COLLIDE2VAugmentedSupConLitModule

# Initialize model
model = COLLIDE2VAugmentedSupConLitModule(
    d_model=512,
    n_heads=8,
    num_layers=4,
    d_ff=256,
    dropout=0.1,
    projection_dim=128,
    temperature=0.07,
    mask_probability=0.2,
    mask_full_particle=False,
    num_augmentations=2,
    use_classification_head=True,
    classification_weight=0.1,
)

# Extract embeddings (after training)
embeddings = model.get_embeddings(x)
```

## Hyperparameter Tuning Guide

### 1. Mask Probability (`mask_probability`)

Controls augmentation strength:

- **0.1**: Mild augmentation, easier task
- **0.2**: Balanced (recommended starting point)
- **0.3-0.5**: Aggressive augmentation, more robust but harder to train

**When to increase**:
- Model is overfitting
- Want more robust representations
- Have sufficient training data

**When to decrease**:
- Training is unstable
- Validation loss not improving
- Limited training data

### 2. Temperature (`temperature`)

Controls hardness of negative mining:

- **0.05**: Very hard negatives, steep gradients
- **0.07**: Balanced (recommended)
- **0.1**: Softer contrasts, more stable

**When to decrease**:
- Classes are very similar (hard classification task)
- Need stronger discrimination
- Training is stable

**When to increase**:
- Training is unstable (loss exploding)
- Gradients are too large
- Model is not converging

### 3. Masking Strategy (`mask_full_particle`)

- **False** (feature-level): More aggressive, good for general robustness
- **True** (particle-level): More realistic, simulates detector effects

**Choose feature-level when**:
- Want maximum augmentation
- Classes differ in subtle features
- Dataset is relatively small

**Choose particle-level when**:
- Want physically realistic augmentation
- Simulating detector inefficiencies
- Have abundant data

### 4. Classification Weight (`classification_weight`)

Balances contrastive and classification objectives:

- **0.0**: Pure contrastive learning (no supervision)
- **0.1**: Light supervision (recommended)
- **0.3-0.5**: Strong supervision (risk of overfitting)

**Recommended**: Start with 0.1 for monitoring, can remove (set to 0.0) after validation

## Monitoring Training

### Key Metrics to Watch

1. **`train/con_loss`**: Contrastive loss (should decrease)
2. **`val/con_loss`**: Validation contrastive loss (monitor for overfitting)
3. **`train/acc`** (if classification head enabled): Training accuracy
4. **`val/acc`**: Validation accuracy
5. **`debug/emb_std`**: Embedding standard deviation (should be > 0.1)
6. **`debug/proj_std`**: Projection standard deviation (should be > 0.1)

### Good Training Signs

✅ Contrastive loss decreasing steadily  
✅ Validation loss following training loss  
✅ Embedding std > 0.1 (not collapsed)  
✅ Validation accuracy improving (if using classification head)  

### Warning Signs

⚠️ **Embedding collapse**: `debug/emb_std` < 0.01
- Solution: Increase temperature, decrease learning rate

⚠️ **Loss not decreasing**: Stuck at high value
- Solution: Decrease mask_probability, increase temperature

⚠️ **Overfitting**: Large gap between train and val loss
- Solution: Increase mask_probability, add more dropout

⚠️ **Unstable training**: Loss oscillating wildly
- Solution: Decrease learning rate, increase temperature

## Comparison with Vanilla SupCon

### When to Use Aug-SupCon

✅ Need robust representations  
✅ Downstream task has missing/corrupted data  
✅ Want to simulate detector inefficiencies  
✅ Have computational resources for 2x forward passes  

### When to Use Vanilla SupCon

✅ Data is clean and complete  
✅ Limited computational budget  
✅ Faster training required  
✅ No need for augmentation robustness  

## Advanced Usage

### Custom Augmentation

You can modify the augmentation strategy by creating a custom augmentation class:

```python
class CustomAugmentation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your custom augmentation logic
        augmented = x + torch.randn_like(x) * 0.1  # Example: add noise
        return augmented
```

### Transfer Learning

After training, extract embeddings for downstream tasks:

```python
# Load trained model
model = COLLIDE2VAugmentedSupConLitModule.load_from_checkpoint(
    checkpoint_path="path/to/checkpoint.ckpt"
)
model.eval()

# Extract embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(input_data)

# Use embeddings for:
# - Classification with a new head
# - Clustering
# - Anomaly detection
# - Retrieval
```

## References

1. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
2. **Supervised Contrastive Learning**: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
3. **Data Augmentation for Particle Physics**: Various papers on detector simulation and robustness

## Troubleshooting

### Issue: "RuntimeError: No trainable parameters found"
**Solution**: Check that model.setup() was called. This should happen automatically in Lightning.

### Issue: "Loss is NaN"
**Solution**: 
- Reduce learning rate
- Increase temperature
- Check for zero-valued embeddings
- Add gradient clipping

### Issue: "Out of memory"
**Solutions**:
- Reduce batch size
- Reduce num_augmentations to 1
- Use gradient checkpointing
- Reduce model size (d_model, num_layers)

### Issue: "Training is very slow"
**Solutions**:
- Set `compile=true` (PyTorch 2.0+)
- Reduce num_augmentations
- Use mixed precision (`trainer.precision=bf16-mixed`)
- Increase batch size (if memory allows)

## Citation

If you use this code, please cite:

```bibtex
@software{collide2v_aug_supcon,
  title={Augmented Supervised Contrastive Learning for Particle Physics},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```
