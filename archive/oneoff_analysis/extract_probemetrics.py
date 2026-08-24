import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Path to the probe evaluation logs
log_dir = "logs/train/runs/2026-02-20_16-19-01/probe_evaluation_baseline/linear_probe"

# Class names mapping (6-class setup)
CLASS_NAMES = {
    '0': 'QCD_inclusive',
    '1': 'DY_to_ll',
    '2': 'Z_to_bb',
    '3': 'W_to_lv',
    '4': 'gamma',
    '5': 'tt_all-lept'
}

# Load TensorBoard events
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("=" * 60)
print("PROBE BASELINE METRICS")
print("=" * 60)

# Get all scalar tags
tags = ea.Tags()['scalars']
print(f"\nAvailable metrics: {len(tags)}")

# Extract key metrics
metrics_dict = {}
for tag in tags:
    events = ea.Scalars(tag)
    if events:
        # Get the last (best) value
        last_value = events[-1].value
        metrics_dict[tag] = last_value

# Display metrics organized by category
print("\n" + "=" * 60)
print("LINEAR PROBE PERFORMANCE")
print("=" * 60)

# Main metrics
main_metrics = ['linear_probe_accuracy', 'linear_probe_f1_macro', 'linear_probe_auroc_macro']
for metric in main_metrics:
    if metric in metrics_dict:
        print(f"{metric:40s}: {metrics_dict[metric]:.4f}")

# Per-class metrics
print("\n" + "-" * 60)
print("PER-CLASS F1 SCORES")
print("-" * 60)
f1_metrics = {k: v for k, v in metrics_dict.items() if 'f1_class_' in k}
for metric, value in sorted(f1_metrics.items()):
    class_idx = metric.split('_')[-1]
    class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
    print(f"{class_name:20s}: {value:.4f}")

print("\n" + "-" * 60)
print("PER-CLASS AUROC SCORES")
print("-" * 60)
auroc_metrics = {k: v for k, v in metrics_dict.items() if 'auroc_class_' in k}
for metric, value in sorted(auroc_metrics.items()):
    class_idx = metric.split('_')[-1]
    class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
    print(f"{class_name:20s}: {value:.4f}")

# KNN metrics if available
print("\n" + "=" * 60)
print("KNN PROBE PERFORMANCE (if available)")
print("=" * 60)
knn_metrics = {k: v for k, v in metrics_dict.items() if 'knn_probe' in k and 'accuracy' in k}
for metric, value in sorted(knn_metrics.items()):
    print(f"{metric:40s}: {value:.4f}")

# Save to CSV
df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['value'])
df.index.name = 'metric'
output_file = os.path.join(log_dir, "extracted_metrics.csv")
df.to_csv(output_file)
print(f"\n✅ Metrics saved to: {output_file}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total metrics logged: {len(metrics_dict)}")
if 'linear_probe_accuracy' in metrics_dict:
    acc = metrics_dict['linear_probe_accuracy']
    print(f"🎯 Linear Probe Accuracy: {acc:.2%}")
if 'linear_probe_auroc_macro' in metrics_dict:
    auroc = metrics_dict['linear_probe_auroc_macro']
    print(f"📊 Macro AUROC: {auroc:.4f}")