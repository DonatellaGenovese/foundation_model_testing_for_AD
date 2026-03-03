# UMAP Visualization for Checkpoints

This script generates UMAP visualizations of learned embeddings from trained checkpoints.

## Quick Start

### Basic Usage

```bash
python visualize_checkpoint.py ckpt_path=/path/to/checkpoint.ckpt
```

### With Custom Parameters

```bash
# Change UMAP parameters
python visualize_checkpoint.py \
    ckpt_path=/path/to/checkpoint.ckpt \
    n_neighbors=20 \
    min_dist=0.2 \
    max_samples=5000

# Specify output directory
python visualize_checkpoint.py \
    ckpt_path=/path/to/checkpoint.ckpt \
    output_dir=./my_visualizations

# Custom title and filename
python visualize_checkpoint.py \
    ckpt_path=/path/to/checkpoint.ckpt \
    title="My Custom Title" \
    output_filename="my_umap.png"
```

### Using Specific Experiment Config

```bash
# Match the experiment configuration used during training
python visualize_checkpoint.py \
    experiment=aug_supcon_6class \
    ckpt_path=/path/to/checkpoint.ckpt
```

### Override Paths Configuration

```bash
# If you get path errors, override the paths config
python visualize_checkpoint.py \
    paths=fm_testing_cineca \
    ckpt_path=/path/to/checkpoint.ckpt

# Override the data label to match your preprocessed data folder
python visualize_checkpoint.py \
    ckpt_path=/path/to/checkpoint.ckpt \
    data.label="v2_6class_highlevel"

# Or use a different paths config
python visualize_checkpoint.py \
    paths=fm_testing_4090 \
    ckpt_path=/path/to/checkpoint.ckpt
```

## Example

```bash
python visualize_checkpoint.py \
    ckpt_path=/leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD/logs/train/runs/2026-02-26_12-00-10/checkpoints/epoch_009_val_loss_0.0044.ckpt
```

## Configuration Parameters

- `ckpt_path`: (Required) Path to checkpoint file
- `data.label`: Data label matching your preprocessed data folder (default: "v2_6class_highlevel")
- `paths`: Paths configuration (default: fm_testing_cineca, options: fm_testing, fm_testing_4090, fm_testing_A6000, fm_testing_cineca)
- `experiment`: Experiment configuration to use (optional, will auto-load from checkpoint if not specified)
- `output_dir`: Output directory for visualization (default: checkpoint_dir/visualizations)
- `output_filename`: Name of output file (default: umap_embeddings.png)
- `title`: Custom plot title (default: auto-generated from checkpoint name)
- `n_neighbors`: UMAP n_neighbors parameter (default: 15)
- `min_dist`: UMAP min_dist parameter (default: 0.1)
- `metric`: Distance metric for UMAP (default: cosine)
- `max_samples`: Maximum samples to visualize (default: 10000)
- `seed`: Random seed for reproducibility (default: 42)

### Important Note on Data Paths

The path to preprocessed data is constructed as: `{paths.eos_preproc_dir}/{data.label}/preprocessed/`

Make sure `data.label` matches your actual preprocessed data folder name. Check your directory:
```bash
ls /leonardo_scratch/fast/IscrC_ELLIF-HE/donatellagenovese/CollideMini_processed/
```

Common labels:
- `v2_6class_highlevel` (6-class classification)
- `v2_15class_highlevel` (15-class classification)
- `QCD_ggHbb_upscaled` (binary classification)

## Output

The script generates a high-resolution PNG file (300 DPI) with:
- UMAP 2D projection of embeddings
- Color-coded by class
- Legend with class names and sample counts
- Saved to `<checkpoint_dir>/visualizations/umap_embeddings.png` by default
