"""
Quick test script for Augmented SupCon module.

This script performs a sanity check on the augmented supervised contrastive learning module
to ensure all components work correctly before full training.

Usage:
    python test_augmented_supcon.py
"""

import torch
import torch.nn as nn
from src.models.collide2v_augmented_supcon import (
    RandomMaskingAugmentation,
    ProjectionHead,
    SimCLRLoss,
)


def test_random_masking_augmentation():
    """Test random masking augmentation module."""
    print("=" * 80)
    print("Testing RandomMaskingAugmentation...")
    print("=" * 80)
    
    # Create a simple feature map
    feature_map = {
        "jets": {
            "start": 0,
            "end": 15,  # 3 features * 5 jets
            "columns": ["pt", "eta", "phi"],
            "topk": 5,
            "count": True,
        },
        "met": {
            "start": 15,
            "end": 17,  # 2 features
            "columns": ["met", "phi"],
            "topk": None,
            "count": False,
        }
    }
    
    # Test feature-level masking
    aug_feature = RandomMaskingAugmentation(
        feature_map=feature_map,
        mask_probability=0.3,
        mask_full_particle=False,
    )
    
    # Create dummy input
    batch_size = 4
    feature_dim = 17
    x = torch.randn(batch_size, feature_dim)
    
    # Apply augmentation
    x_aug = aug_feature(x)
    
    # Check shape
    assert x_aug.shape == x.shape, f"Shape mismatch: {x_aug.shape} != {x.shape}"
    
    # Check that some values are masked (set to 0)
    num_masked = (x_aug == 0.0).sum().item()
    num_originally_zero = (x == 0.0).sum().item()
    assert num_masked > num_originally_zero, "No masking occurred!"
    
    print(f"✅ Feature-level masking: {num_masked - num_originally_zero} elements masked")
    
    # Test particle-level masking
    aug_particle = RandomMaskingAugmentation(
        feature_map=feature_map,
        mask_probability=0.3,
        mask_full_particle=True,
    )
    
    x_aug_particle = aug_particle(x)
    assert x_aug_particle.shape == x.shape, "Shape mismatch for particle-level masking"
    
    print(f"✅ Particle-level masking: Working correctly")
    print()


def test_projection_head():
    """Test projection head module."""
    print("=" * 80)
    print("Testing ProjectionHead...")
    print("=" * 80)
    
    # Create projection head
    proj_head = ProjectionHead(
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
    )
    
    # Create dummy embeddings
    batch_size = 16
    embeddings = torch.randn(batch_size, 512)
    
    # Forward pass
    projections = proj_head(embeddings)
    
    # Check shape
    assert projections.shape == (batch_size, 128), f"Shape mismatch: {projections.shape}"
    
    # Check that it's learnable (has gradients)
    loss = projections.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in proj_head.parameters())
    assert has_grad, "Projection head has no gradients!"
    
    print(f"✅ Projection head: Input {embeddings.shape} → Output {projections.shape}")
    print()


def test_simclr_loss():
    """Test SimCLR loss function."""
    print("=" * 80)
    print("Testing SimCLRLoss...")
    print("=" * 80)
    
    # Create loss module
    criterion = SimCLRLoss(temperature=0.07)
    
    # Create dummy features (L2-normalized)
    batch_size = 16
    embedding_dim = 128
    features = torch.randn(batch_size, embedding_dim)
    features = features / features.norm(dim=1, keepdim=True)  # L2 normalize
    
    # Create dummy labels (4 classes)
    labels = torch.randint(0, 4, (batch_size,))
    
    # Compute loss
    loss = criterion(features, labels)
    
    # Check that loss is a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    
    # Check that loss is positive
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    
    # Check that loss is finite
    assert torch.isfinite(loss), "Loss is not finite!"
    
    # Test with gradients
    loss.backward()
    assert features.grad is not None, "No gradients computed!"
    
    print(f"✅ SimCLR Loss: {loss.item():.4f}")
    print(f"✅ Gradients: max={features.grad.abs().max().item():.4f}")
    print()
    
    # Test edge case: all same class (should still work but may give warning in actual training)
    try:
        same_class_labels = torch.zeros(batch_size, dtype=torch.long)
        loss_same = criterion(features, same_class_labels)
        print(f"✅ Single-class batch: Loss = {loss_same.item():.4f}")
    except Exception as e:
        print(f"⚠️  Single-class batch raised exception (expected in some cases): {e}")
    
    print()


def test_augmentation_views():
    """Test creating multiple augmented views."""
    print("=" * 80)
    print("Testing Multiple Augmented Views...")
    print("=" * 80)
    
    # Simple feature map
    feature_map = {
        "features": {
            "start": 0,
            "end": 10,
            "columns": ["f1", "f2"],
            "topk": 5,
            "count": False,
        }
    }
    
    aug = RandomMaskingAugmentation(
        feature_map=feature_map,
        mask_probability=0.2,
        mask_full_particle=False,
    )
    
    # Create input
    batch_size = 8
    x = torch.randn(batch_size, 10)
    
    # Create 2 views
    view1 = aug(x)
    view2 = aug(x)
    
    # Check that views are different
    assert not torch.allclose(view1, view2), "Views should be different!"
    
    # Concatenate views (as done in training)
    x_concat = torch.cat([view1, view2], dim=0)
    labels = torch.randint(0, 3, (batch_size,))
    labels_repeated = labels.repeat(2)
    
    assert x_concat.shape[0] == batch_size * 2, "Concatenation failed"
    assert labels_repeated.shape[0] == batch_size * 2, "Label repetition failed"
    
    print(f"✅ View 1 shape: {view1.shape}")
    print(f"✅ View 2 shape: {view2.shape}")
    print(f"✅ Concatenated shape: {x_concat.shape}")
    print(f"✅ Labels repeated: {labels.shape} → {labels_repeated.shape}")
    print()


def test_full_forward_pass():
    """Test a complete forward pass through the augmentation pipeline."""
    print("=" * 80)
    print("Testing Full Forward Pass (Augmentation + Projection + Loss)...")
    print("=" * 80)
    
    # Setup
    feature_map = {
        "group1": {
            "start": 0,
            "end": 20,
            "columns": ["f1", "f2", "f3", "f4"],
            "topk": 5,
            "count": False,
        }
    }
    
    batch_size = 16
    feature_dim = 20
    d_model = 128
    projection_dim = 64
    num_classes = 4
    
    # Create components
    aug = RandomMaskingAugmentation(feature_map, mask_probability=0.2)
    
    # Simulate encoder (simple linear layer)
    encoder = nn.Linear(feature_dim, d_model)
    
    proj_head = ProjectionHead(d_model, hidden_dim=128, output_dim=projection_dim)
    
    criterion = SimCLRLoss(temperature=0.07)
    
    # Create data
    x = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Full pipeline
    # 1. Create augmented views
    view1 = aug(x)
    view2 = aug(x)
    x_concat = torch.cat([view1, view2], dim=0)
    labels_repeated = labels.repeat(2)
    
    # 2. Encode
    embeddings = encoder(x_concat)
    
    # 3. Project
    projections = proj_head(embeddings)
    projections = projections / projections.norm(dim=1, keepdim=True)  # L2 normalize
    
    # 4. Compute loss
    loss = criterion(projections, labels_repeated)
    
    # 5. Backward pass
    loss.backward()
    
    print(f"✅ Input: {x.shape}")
    print(f"✅ Augmented (2 views): {x_concat.shape}")
    print(f"✅ Embeddings: {embeddings.shape}")
    print(f"✅ Projections (normalized): {projections.shape}")
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Backward pass: Successful")
    
    # Check gradients
    encoder_has_grad = any(p.grad is not None for p in encoder.parameters())
    proj_has_grad = any(p.grad is not None for p in proj_head.parameters())
    
    assert encoder_has_grad, "Encoder has no gradients!"
    assert proj_has_grad, "Projection head has no gradients!"
    
    print(f"✅ Encoder gradients: Present")
    print(f"✅ Projection gradients: Present")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "AUGMENTED SUPCON SANITY CHECKS" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        test_random_masking_augmentation()
        test_projection_head()
        test_simclr_loss()
        test_augmentation_views()
        test_full_forward_pass()
        
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 28 + "ALL TESTS PASSED! ✅" + " " * 30 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print("The Augmented SupCon module is ready to use!")
        print("Run training with: python src/train.py experiment=aug_supcon_6class")
        print()
        
    except Exception as e:
        print()
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 30 + "TEST FAILED! ❌" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
