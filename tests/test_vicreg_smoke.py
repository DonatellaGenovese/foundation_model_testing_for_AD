"""
Smoke test for COLLIDE2VVICRegLitModule and COLLIDE2VVCRegLitModule.

Tests components and a full forward/loss pass with synthetic data —
no real data or datamodule required.
"""

import functools
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD")

from src.models.collide2v_vicreg import (
    RandomMaskingAugmentation,
    ProjectionHead,
    VICRegLoss,
    VCRegLoss,
    COLLIDE2VVICRegLitModule,
    COLLIDE2VVCRegLitModule,
)

# ---------------------------------------------------------------------------
# Minimal synthetic feature map (2 groups: jets + met)
# ---------------------------------------------------------------------------
FEATURE_MAP = {
    "jets": {
        "start": 0,
        "end":   24,
        "topk":  4,
        "columns": ["pt", "eta", "phi", "mass", "btag", "charge"],
        "count": False,
    },
    "met": {
        "start": 24,
        "end":   26,
        "topk":  None,
        "columns": ["met", "phi"],
        "count": False,
    },
}

FEATURE_DIM = 26
BATCH       = 32
NUM_CLASSES = 3
D_MODEL     = 64


def make_batch():
    x      = torch.randn(BATCH, FEATURE_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    return x, labels


def make_partial_optimizer():
    return functools.partial(torch.optim.AdamW, lr=1e-3)


# ---------------------------------------------------------------------------
# 1. RandomMaskingAugmentation
# ---------------------------------------------------------------------------
def test_augmentation():
    aug = RandomMaskingAugmentation(FEATURE_MAP, mask_probability=0.3, mask_full_particle=False)
    x = torch.randn(BATCH, FEATURE_DIM)
    out = aug(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert not torch.equal(out, x), "Augmentation had no effect"
    print("  [PASS] RandomMaskingAugmentation (feature-level)")

    aug_full = RandomMaskingAugmentation(FEATURE_MAP, mask_probability=0.5, mask_full_particle=True)
    out_full = aug_full(x)
    assert out_full.shape == x.shape
    print("  [PASS] RandomMaskingAugmentation (particle-level)")


# ---------------------------------------------------------------------------
# 2. ProjectionHead
# ---------------------------------------------------------------------------
def test_projection_head():
    head = ProjectionHead(input_dim=D_MODEL, hidden_dim=128, output_dim=32)
    x = torch.randn(BATCH, D_MODEL)
    out = head(x)
    assert out.shape == (BATCH, 32), f"Shape mismatch: {out.shape}"
    print("  [PASS] ProjectionHead")


# ---------------------------------------------------------------------------
# 3. VICRegLoss
# ---------------------------------------------------------------------------
def test_vicreg_loss():
    criterion = VICRegLoss(inv_weight=25.0, var_weight=25.0, cov_weight=1.0)
    z      = torch.randn(BATCH, 32)
    z_prime = torch.randn(BATCH, 32)
    out = criterion(z, z_prime)
    assert "loss" in out and "inv_loss" in out and "var_loss" in out and "cov_loss" in out
    assert out["loss"].shape == (), f"Loss not scalar: {out['loss'].shape}"
    assert not torch.isnan(out["loss"]), "NaN in VICReg loss"
    # Identical inputs → inv_loss should be 0
    out_same = criterion(z, z)
    assert out_same["inv_loss"].item() < 1e-6, "inv_loss not zero for identical inputs"
    print("  [PASS] VICRegLoss")


# ---------------------------------------------------------------------------
# 4. VCRegLoss
# ---------------------------------------------------------------------------
def test_vcreg_loss():
    criterion = VCRegLoss(var_weight=25.0, cov_weight=1.0)
    h = torch.randn(BATCH, D_MODEL)
    out = criterion(h)
    assert "loss" in out and "var_loss" in out and "cov_loss" in out
    assert out["loss"].shape == (), f"Loss not scalar: {out['loss'].shape}"
    assert not torch.isnan(out["loss"]), "NaN in VCReg loss"
    print("  [PASS] VCRegLoss")


# ---------------------------------------------------------------------------
# 5. COLLIDE2VVICRegLitModule — forward + model_step
# ---------------------------------------------------------------------------
def test_vicreg_module():
    model = COLLIDE2VVICRegLitModule(
        d_model=D_MODEL,
        n_heads=2,
        num_layers=1,
        d_ff=128,
        dropout=0.0,
        projection_dim=32,
        hidden_projection_dim=64,
        inv_weight=25.0,
        var_weight=25.0,
        cov_weight=1.0,
        mask_probability=0.2,
        mask_full_particle=False,
        use_classification_head=True,
        classification_weight=0.1,
        optimizer=make_partial_optimizer(),
    )
    model._build_model_components(FEATURE_MAP, NUM_CLASSES)

    x, labels = make_batch()

    # forward
    model.eval()
    with torch.no_grad():
        emb, proj, logits = model(x)
    assert emb.shape   == (BATCH, D_MODEL), f"Embedding shape: {emb.shape}"
    assert proj.shape  == (BATCH, 32),      f"Projection shape: {proj.shape}"
    assert logits.shape == (BATCH, NUM_CLASSES)
    print("  [PASS] VICReg forward (eval)")

    # model_step with augmentation (train)
    model.train()
    out = model.model_step((x, labels), apply_augmentation=True)
    assert not torch.isnan(out["vicreg_loss"]), "NaN in train vicreg_loss"
    assert out["inv_loss"].item() >= 0
    print("  [PASS] VICReg model_step (augmentation=True)")

    # model_step without augmentation (val) — single forward pass
    model.eval()
    with torch.no_grad():
        out_val = model.model_step((x, labels), apply_augmentation=False)
    assert out_val["inv_loss"].item() < 1e-6, \
        f"inv_loss not ~0 at val: {out_val['inv_loss'].item()}"
    print("  [PASS] VICReg model_step (augmentation=False, inv_loss≈0)")


# ---------------------------------------------------------------------------
# 6. COLLIDE2VVCRegLitModule — forward + model_step
# ---------------------------------------------------------------------------
def test_vcreg_module():
    model = COLLIDE2VVCRegLitModule(
        d_model=D_MODEL,
        n_heads=2,
        num_layers=1,
        d_ff=128,
        dropout=0.0,
        var_weight=25.0,
        cov_weight=1.0,
        optimizer=make_partial_optimizer(),
    )
    model._build_model_components(FEATURE_MAP, NUM_CLASSES)

    x, labels = make_batch()

    # forward
    model.eval()
    with torch.no_grad():
        emb, logits = model(x)
    assert emb.shape    == (BATCH, D_MODEL),    f"Embedding shape: {emb.shape}"
    assert logits.shape == (BATCH, NUM_CLASSES), f"Logits shape: {logits.shape}"
    print("  [PASS] VCReg forward (eval)")

    # model_step
    model.train()
    out = model.model_step((x, labels))
    assert not torch.isnan(out["loss"]),       "NaN in total loss"
    assert not torch.isnan(out["ce_loss"]),    "NaN in CE loss"
    assert not torch.isnan(out["vcreg_loss"]), "NaN in VCReg loss"
    assert out["preds"].shape == (BATCH,)
    print("  [PASS] VCReg model_step")


# ---------------------------------------------------------------------------
# 7. get_embeddings consistency
# ---------------------------------------------------------------------------
def test_get_embeddings():
    for cls, name, kwargs in [
        (COLLIDE2VVICRegLitModule, "VICReg", dict(
            d_model=D_MODEL, n_heads=2, num_layers=1, d_ff=128, dropout=0.0,
            projection_dim=32, hidden_projection_dim=64,
            use_classification_head=False,
            optimizer=make_partial_optimizer(),
        )),
        (COLLIDE2VVCRegLitModule, "VCReg", dict(
            d_model=D_MODEL, n_heads=2, num_layers=1, d_ff=128, dropout=0.0,
            optimizer=make_partial_optimizer(),
        )),
    ]:
        model = cls(**kwargs)
        model._build_model_components(FEATURE_MAP, NUM_CLASSES)
        model.eval()
        x, _ = make_batch()
        with torch.no_grad():
            emb = model.get_embeddings(x)
        assert emb.shape == (BATCH, D_MODEL), f"{name} get_embeddings shape: {emb.shape}"
        print(f"  [PASS] {name} get_embeddings")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("RandomMaskingAugmentation",        test_augmentation),
        ("ProjectionHead",                   test_projection_head),
        ("VICRegLoss",                       test_vicreg_loss),
        ("VCRegLoss",                        test_vcreg_loss),
        ("COLLIDE2VVICRegLitModule",         test_vicreg_module),
        ("COLLIDE2VVCRegLitModule",          test_vcreg_module),
        ("get_embeddings consistency",       test_get_embeddings),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
