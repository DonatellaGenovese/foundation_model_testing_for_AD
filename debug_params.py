"""
Debug script to check if parameters are properly registered
"""
import torch
import torch.nn as nn
from lightning import LightningModule

class TestModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = None  # Set to None initially
        
    def setup(self, stage):
        # Initialize in setup (like SupCon model does)
        self.layer1 = nn.Linear(10, 10)
        print(f"After setup: layer1 is {type(self.layer1)}")
        
    def configure_optimizers(self):
        params = list(self.parameters())
        print(f"Number of parameter tensors in configure_optimizers: {len(params)}")
        if len(params) > 0:
            print(f"First param shape: {params[0].shape}")
        else:
            print("WARNING: No parameters found!")
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Test
print("=== Testing parameter registration ===")
model = TestModule()

# Simulate Lightning's lifecycle
model.setup(stage='fit')
optimizer_config = model.configure_optimizers()

print(f"\n=== After manual setup ===")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Now test with Lightning Trainer
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

print("\n=== Testing with Lightning Trainer ===")

class DummyDataModule(torch.nn.Module):
    pass

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 10)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=10)

model2 = TestModule()

# Check params before trainer
print(f"Before trainer: {sum(p.numel() for p in model2.parameters())} params")

trainer = Trainer(max_epochs=1, accelerator='cpu', logger=False, enable_checkpointing=False)

# Attach a simple forward and training_step
def forward(self, x):
    if self.layer1 is not None:
        return self.layer1(x)
    return x

def training_step(self, batch, batch_idx):
    x, y = batch
    pred = self(x)
    loss = ((pred - y) ** 2).mean()
    return loss

TestModule.forward = forward
TestModule.training_step = training_step

# This will call setup internally
# trainer.fit(model2, train_loader)

print("\nConclusion: If parameters show 0 before trainer.fit but >0 after setup,")
print("then the issue is with when configure_optimizers is called")
