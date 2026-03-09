# Anomaly Detection with Autoencoders

## 🎯 Overview

This document explains how to use the anomaly detection framework in this repository.

### Two Strategies
1. **Raw Features** (Baseline): Train autoencoder directly on input features
2. **Embeddings** (Advanced): Train autoencoder on learned representations from a pretrained encoder (SupCon/contrastive learning)


## 📁 Configuration Files

### Base Config
**`configs/anomaly_detection.yaml`**
- Main configuration file with default settings
- Defines trainer, model, data, and paths settings
- Override via experiment configs or command line

### Experiment Configs (`configs/experiment/`)

| Config File | Purpose | Use For |
|------------|---------|---------|
| `anomaly_qcd_vs_higgs_raw.yaml` | Baseline with raw features | Quick baseline, no pretrained model needed |
| `anomaly_qcd_vs_higgs_embedding.yaml` | Embeddings from pretrained encoder | Best performance, requires trained encoder |

### Key Parameters in Experiment Configs

```yaml
# Mode: 'raw' or 'embeddings'
mode: "raw"  # or "embeddings"

# Normal class (train only on these classes)
normal_classes: [0]  # define the list of classes 

# Anomaly classes (detect these)
anomaly_classes: [15, 17]  # define the list of anomalies

# For embeddings mode:
ckpt_path: "/path/to/encoder.ckpt"  # Pretrained encoder checkpoint
model_class: "src.models.xxx.XXXLitModule"  # Encoder model class
use_projections: true  # Use projection head (true) or encoder (false)

# Autoencoder architecture
model:
  compression: 16  # Compression ratio (input_dim / bottleneck_dim)
  depth: 3  # Number of hidden layers
  dropout: 0.1
  lr: 0.001
  weight_decay: 1e-5

# Data configuration
data:
  batch_size: 512
  train_val_test_split_per_class: [1000, 200, 200]
  label: "v2_18class_highlevel"
```

---

### Approach 1: Raw Features (Baseline)

**When to use:**
- Quick baseline without pretrained models
- Want to understand raw feature performance
- Don't have a trained contrastive encoder yet

**Workflow:**
```
Raw Features → Autoencoder → Reconstruction Error → Anomaly Score
```

### Approach 2: Embeddings 
**When to use:**
- Have a pretrained SupCon/contrastive encoder
- Want best anomaly detection performance
- Embeddings capture learned physics patterns

**Workflow:**
```
Raw Features → Pretrained Encoder → Embeddings → Autoencoder → Reconstruction Error → Anomaly Score
```
---

### 1. Raw Features Approach

```bash
# Train autoencoder on raw features
python src/train_anomaly_detection.py \
    experiment=anomaly_qcd_vs_higgs_raw \
    seed=42

# Or submit SLURM job
sbatch run_ae_qcd_vs_higgs_raw.sh
```

### 2. Embeddings Approach

**Step 1: Train a contrastive encoder (if not done yet)**
```bash
# Train SupCon model first
python src/train.py experiment=vanillasupcon_15class_pretrain
```

**Step 2: Train autoencoder on embeddings**
```bash
# Update ckpt_path in config, then run:
python src/train_anomaly_embedding.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/path/to/your/checkpoint.ckpt \
    seed=42

# Or submit SLURM job (after updating checkpoint path in script)
sbatch run_ae_qcd_vs_higgs_embedding.sh
```

---

## 📖 Detailed Usage

### Script Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train_anomaly_detection.py` | Train AE on raw features | Raw data | Trained autoencoder |
| `train_anomaly_embedding.py` | Train AE on embeddings | Embeddings + checkpoint | Trained autoencoder |
| `extract_embeddings.py` | Extract embeddings from encoder | Checkpoint + data | Embeddings (.npz files) |

### Training on Raw Features

```bash
python src/train_anomaly_detection.py \
    experiment=anomaly_qcd_vs_higgs_raw \
    normal_classes=[0] \
    anomaly_classes=[15,17] \
    model.compression=16 \
    model.depth=3 \
    model.dropout=0.1 \
    trainer.max_epochs=50 \
    seed=42
```

**What happens:**
1. Loads raw input features from data
2. Filters to keep only normal (QCD) + anomaly classes (Higgs)
3. Trains autoencoder **ONLY on normal class** (QCD)
4. Validates on QCD + Higgs to monitor separation
5. Tests on QCD + Higgs to evaluate anomaly detection
6. Saves plots and metrics

### Training on Embeddings

```bash
python src/train_anomaly_embedding.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/path/to/encoder.ckpt \
    use_projections=true \
    model.compression=4 \
    model.depth=1 \
    seed=42
```

**What happens:**
1. Checks if embeddings exist; if not, calls `extract_embeddings.py`
2. Loads embeddings from disk (train/val/test splits)
3. Filters to keep normal + anomaly classes
4. Trains autoencoder **ONLY on normal embeddings**
5. Tests on normal + anomaly embeddings
6. Saves plots and metrics

### Extracting Embeddings Manually

```bash
python src/extract_embeddings.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/path/to/encoder.ckpt \
    embedding_output_dir=./embeddings \
    use_projections=true \
    seed=42
```

**Outputs:**
- `embeddings/train_embeddings.npz` - Training embeddings
- `embeddings/val_embeddings.npz` - Validation embeddings
- `embeddings/test_embeddings.npz` - Test embeddings
- `embeddings/metadata.json` - Metadata about extraction

---

## 🔧 Advanced Configuration

### Understanding Autoencoder Parameters

The autoencoder uses two key parameters:

**`compression`** (int): Compression ratio
- Defines: `bottleneck_dim = input_dim / compression`
- Higher value = more compression = smaller bottleneck
- Example: `compression=16` with 512D input → 32D bottleneck

**`depth`** (int): Number of hidden layers
- Defines how many layers between input and bottleneck
- Layers are automatically generated with progressive compression
- Example: `depth=3` with 512D input → `512 → 256 → 128 → 64 → 32 (bottleneck)`

**Architecture Flow:**
```
Input (512D)
   ↓ Layer 1 (512 → 256)
   ↓ Layer 2 (256 → 128)
   ↓ Layer 3 (128 → 64)
   ↓ Bottleneck Layer (64 → 32)
   ↓ Layer 3 (32 → 64)
   ↓ Layer 2 (64 → 128)
   ↓ Layer 1 (128 → 256)
Output (256 → 512)
```

### Customizing Normal and Anomaly Classes

Edit the experiment config or override via command line:

```yaml
# In config file:
normal_classes: [0]  # QCD_inclusive
anomaly_classes: [15, 17]  # VBFHbb, ggHtautau

# Or command line:
python src/train_anomaly_detection.py \
    normal_classes=[0,13] \
    anomaly_classes=[15,17]
```

### Adjusting Autoencoder Architecture

**How to choose parameters:**

1. **`compression`**: Controls bottleneck size
   - Raw features: 8-32 (more compression needed)
   - Embeddings: 2-8 (less compression needed)
   - Higher = smaller bottleneck = more compression

2. **`depth`**: Controls network complexity
   - Raw features: 3-4 (deeper network)
   - Embeddings: 1-2 (simpler network)
   - More depth = more gradual compression

**For raw features (high dimensional input):**
```yaml
model:
  compression: 16  # Higher compression ratio
  depth: 3  # Deeper network
  dropout: 0.1
```

**For embeddings (already compressed):**
```yaml
model:
  compression: 4  # Lower compression ratio
  depth: 1  # Simpler architecture
  dropout: 0.2
```

### Migration from Old Config Format

If you have old configs with `bottleneck_dim` and `hidden_dims`:

```yaml
# ❌ OLD FORMAT (no longer supported)
model:
  bottleneck_dim: 32
  hidden_dims: [256, 128, 64]

# ✅ NEW FORMAT (required)
model:
  compression: 16  # input_dim / 32 ≈ 16 (for 512D input)
  depth: 3  # 3 hidden layers
```

**Conversion guide:**
- `depth` = number of elements in `hidden_dims`
- `compression` = `input_dim / bottleneck_dim` (approximate)

### Using Different Encoders

For embeddings approach, you can use any trained encoder:

```bash
# Vanilla SupCon encoder
python src/train_anomaly_embedding.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/path/to/vanilla_supcon.ckpt \
    model_class=src.models.collide2v_vanillasupcon.COLLIDE2VVanillaSupConLitModule \
    use_projections=true

# Augmented SupCon encoder
python src/train_anomaly_embedding.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/path/to/augmented_supcon.ckpt \
    model_class=src.models.collide2v_augmented_supcon.COLLIDE2VAugmentedSupConLitModule \
    use_projections=false  # Use encoder embeddings instead
```

### Choosing Encoder vs Projection Embeddings

For SupCon models, you have two options:

1. **Projection Head Embeddings** (`use_projections=true`)
   - Lower dimensional (e.g., 128D)
   - Optimized for contrastive loss
   - Often better for anomaly detection

2. **Encoder Embeddings** (`use_projections=false`)
   - Higher dimensional (e.g., 512D)
   - General representations
   - May need deeper autoencoder






