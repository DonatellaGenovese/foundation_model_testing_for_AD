"""
Test gradient accumulation with contrastive loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        similarity_matrix = torch.matmul(features, features.T)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits / self.temperature)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        
        log_prob = logits / self.temperature - torch.log(
            (exp_logits * logits_mask).sum(1, keepdim=True) + 1e-9
        )
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        
        return loss

print("="*60)
print("TESTING GRADIENT ACCUMULATION WITH CONTRASTIVE LOSS")
print("="*60)

# Simulate a full batch of 1024 samples
torch.manual_seed(42)
full_batch_size = 1024
embedding_dim = 128
num_classes = 6

# Create balanced batch: ~170 samples per class (170*6 = 1020, add 4 more)
samples_per_class = full_batch_size // num_classes  # 170
remainder = full_batch_size % num_classes  # 4
labels_full = torch.cat([torch.full((samples_per_class + (1 if i < remainder else 0),), i) for i in range(num_classes)])

features_full = torch.randn(full_batch_size, embedding_dim)
features_full = F.normalize(features_full, dim=-1, p=2)

criterion = SupConLoss(temperature=0.1)

# Method 1: Process full batch at once (correct)
loss_full = criterion(features_full, labels_full)
print(f"\n1. Full batch (1024): loss = {loss_full.item():.4f}")

# Method 2: Simulate gradient accumulation (what Lightning does)
# Split into 2 micro-batches of 512
features_1 = features_full[:512]
features_2 = features_full[512:]
labels_1 = labels_full[:512]
labels_2 = labels_full[512:]

loss_1 = criterion(features_1, labels_1)
loss_2 = criterion(features_2, labels_2)
loss_accumulated = (loss_1 + loss_2) / 2  # Average the losses

print(f"\n2. Gradient accumulation (2 x 512):")
print(f"   Micro-batch 1 loss: {loss_1.item():.4f}")
print(f"   Micro-batch 2 loss: {loss_2.item():.4f}")
print(f"   Average loss: {loss_accumulated.item():.4f}")

print(f"\n" + "="*60)
print(f"DIFFERENCE: {abs(loss_full.item() - loss_accumulated.item()):.4f}")
print("="*60)

if abs(loss_full.item() - loss_accumulated.item()) > 0.01:
    print("\n❌ PROBLEM DETECTED!")
    print("   Gradient accumulation gives DIFFERENT loss than full batch!")
    print("   ")
    print("   Why? Contrastive loss compares samples WITHIN each batch.")
    print("   With grad accumulation:")
    print("   - Micro-batch 1 (512) only compares its 512 samples")
    print("   - Micro-batch 2 (512) only compares its 512 samples")
    print("   - They NEVER compare across micro-batches!")
    print("   ")
    print("   Full batch (1024) compares ALL samples together.")
    print("   ")
    print("   SOLUTION: Remove gradient accumulation for contrastive learning!")
else:
    print("\n✅ No significant difference (unlikely for contrastive loss)")

print("\n" + "="*60)
print("WHAT'S HAPPENING IN YOUR MODEL:")
print("="*60)
print("With accumulate_grad_batches=2 and batch_size=512:")
print("1. Each micro-batch of 512 has ~85 samples per class")
print("2. Contrastive loss only sees these 85 positives")
print("3. The loss can't leverage the full 170 positives from both micro-batches")
print("4. This severely limits the effectiveness of contrastive learning")
print("5. The model can't learn good representations!")
