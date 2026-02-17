"""
Diagnostic script to check SupCon training issues
"""
import torch
import os
import sys
os.chdir("/hdd3/dongen/Desktop/Collide2v/foundation_model_testing_for_AD")
sys.path.insert(0, "/hdd3/dongen/Desktop/Collide2v/foundation_model_testing_for_AD")

# Load the trained checkpoint
#ckpt_path = "/hdd3/dongen/Desktop/Collide2v/foundation_model_testing_for_AD/logs/train/runs/2026-02-17_13-51-25/checkpoints/epoch_epoch=006_val_con_loss_val/con_loss=6.1845.ckpt"
ckpt_path = "/hdd3/dongen/Desktop/Collide2v/foundation_model_testing_for_AD/logs/train/runs/2026-02-16_17-46-40/588720815904048509/a6aae44a567246beaec1800a02566aed/checkpoints/epoch_epoch=006_val_con_loss_val/con_loss=5.7789.ckpt"

print("Loading checkpoint...")
from src.models.collide2v_vanillasupcon import COLLIDE2VVanillaSupConLitModule
import hydra
from omegaconf import DictConfig, OmegaConf

# Load checkpoint (weights_only=False for compatibility with functools.partial)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Check hyperparameters
if 'hyper_parameters' in checkpoint:
    print("\n=== Hyperparameters ===")
    for k, v in checkpoint['hyper_parameters'].items():
        if k not in ['optimizer', 'scheduler']:
            print(f"  {k}: {v}")

# Check if model has parameters
if 'state_dict' in checkpoint:
    print(f"\n=== Model State Dict ===")
    state_dict = checkpoint['state_dict']
    print(f"Total keys in state_dict: {len(state_dict)}")
    
    # Group by component
    encoder_params = [k for k in state_dict.keys() if k.startswith('encoder.')]
    proj_params = [k for k in state_dict.keys() if k.startswith('projection_head.')]
    cls_params = [k for k in state_dict.keys() if k.startswith('classification_head.')]
    
    print(f"Encoder parameters: {len(encoder_params)}")
    print(f"Projection head parameters: {len(proj_params)}")
    print(f"Classification head parameters: {len(cls_params)}")
    
    if len(encoder_params) == 0:
        print("\n⚠️  WARNING: No encoder parameters found!")
        print("This means the model was saved before setup() was called!")
        print("The optimizer has NO PARAMETERS to optimize!")
    
    # Check a few parameter values to see if they're random or trained
    if len(encoder_params) > 0:
        first_encoder_param = state_dict[encoder_params[0]]
        print(f"\nFirst encoder param shape: {first_encoder_param.shape}")
        print(f"First encoder param mean: {first_encoder_param.mean().item():.6f}")
        print(f"First encoder param std: {first_encoder_param.std().item():.6f}")
        
        # Check if params are close to initialization
        if abs(first_encoder_param.mean().item()) < 0.01 and 0.01 < first_encoder_param.std().item() < 0.1:
            print("⚠️  Parameters look like they haven't been updated much from initialization!")
    
    if len(proj_params) > 0:
        first_proj_param = state_dict[proj_params[0]]
        print(f"\nFirst projection head param shape: {first_proj_param.shape}")
        print(f"First projection head param mean: {first_proj_param.mean().item():.6f}")
        print(f"First projection head param std: {first_proj_param.std().item():.6f}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
if len(encoder_params) == 0:
    print("❌ CRITICAL BUG: Model has NO parameters!")
    print("   The model components are initialized in setup() but")
    print("   configure_optimizers() is called BEFORE setup().")
    print("   ")
    print("   SOLUTION: Move model initialization from setup() to __init__()")
    print("   OR use configure_model() hook instead of setup()")
else:
    print("✅ Model has parameters")
    print("   The issue is likely elsewhere (loss function, data, hyperparams)")
