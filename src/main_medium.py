import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

from preprocessing import load_audio
from features_fast import extract_time_features, extract_freq_features
from model_fast import train_model_fast

# ============ CONFIGURATION ============
DATA_PATH = "../data"
MAX_FILES_PER_CLASS = 25  # MEDIUM: 25 files per class (50 total)
# Change to 50 for even more testing
# =======================================

X_time = []
X_freq = []
X_combined = []
y = []

start_time = time.time()

# Load Data
print("="*70)
print("LOADING AUDIO FILES (MEDIUM DATASET MODE)")
print("="*70)
print(f"Using {MAX_FILES_PER_CLASS} files per class ({MAX_FILES_PER_CLASS*2} total)\n")

for label in ["healthy", "parkinson"]:
    folder = os.path.join(DATA_PATH, label)

    if not os.path.exists(folder):
        print(f"❌ Warning: {folder} not found")
        continue

    files = os.listdir(folder)[:MAX_FILES_PER_CLASS]
    print(f"📂 Loading {len(files)} {label.upper()} files...")

    loaded_count = 0
    failed_count = 0
    
    for idx, file in enumerate(files, 1):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder, file)

            try:
                signal, sr = load_audio(file_path)

                time_feat = extract_time_features(signal, sr)
                freq_feat = extract_freq_features(signal, sr)

                X_time.append(time_feat)
                X_freq.append(freq_feat)
                
                # Combine both features
                combined_feat = time_feat + freq_feat
                X_combined.append(combined_feat)

                y.append(0 if label == "healthy" else 1)
                loaded_count += 1
                
                # Progress bar
                progress = int((idx / len(files)) * 20)
                bar = "█" * progress + "░" * (20 - progress)
                print(f"  [{bar}] {idx}/{len(files)} - {file}", end='\r')
                
            except Exception as e:
                failed_count += 1
                continue
    
    print(f"\n  ✓ Successfully loaded {loaded_count}/{len(files)} files")
    if failed_count > 0:
        print(f"  ⚠️  Failed: {failed_count} files\n")
    else:
        print()

# Convert to numpy arrays
X_time = np.array(X_time)
X_freq = np.array(X_freq)
X_combined = np.array(X_combined)
y = np.array(y)

if len(y) < 10:
    print("❌ ERROR: Not enough samples! Need at least 10 samples")
    exit()

print("="*70)
print("📊 DATASET SUMMARY")
print("="*70)
print(f"✓ Total samples:              {len(y)}")
print(f"  - Healthy:                  {np.sum(y == 0)}")
print(f"  - Parkinson's:              {np.sum(y == 1)}")
print(f"  - Class balance:            {np.sum(y == 0) / len(y) * 100:.1f}% vs {np.sum(y == 1) / len(y) * 100:.1f}%")
print(f"\n✓ Time domain features:       {X_time.shape[1]} features × {X_time.shape[0]} samples")
print(f"✓ Frequency domain features:  {X_freq.shape[1]} features × {X_freq.shape[0]} samples")
print(f"✓ Combined features:          {X_combined.shape[1]} features × {X_combined.shape[0]} samples")

# Normalize all feature sets BEFORE training
print("\n" + "="*70)
print("⚙️  NORMALIZING FEATURES")
print("="*70)

scaler_time = StandardScaler()
X_time_normalized = scaler_time.fit_transform(X_time)
print("✓ Time domain normalized")

scaler_freq = StandardScaler()
X_freq_normalized = scaler_freq.fit_transform(X_freq)
print("✓ Frequency domain normalized")

scaler_combined = StandardScaler()
X_combined_normalized = scaler_combined.fit_transform(X_combined)
print("✓ Combined domain normalized")

# Train models
print("\n" + "="*70)
print("🤖 TRAINING MODELS")
print("="*70)

acc_time = train_model_fast(X_time_normalized, y, "Time Domain")
print("\n" + "-"*70 + "\n")

acc_freq = train_model_fast(X_freq_normalized, y, "Frequency Domain")
print("\n" + "-"*70 + "\n")

acc_combined = train_model_fast(X_combined_normalized, y, "Combined (Time + Frequency)")

# Print results
print("\n" + "="*70)
print("⭐ FINAL RESULTS SUMMARY ⭐")
print("="*70)
print(f"Time Domain Accuracy:              {acc_time:.4f} ({acc_time*100:.2f}%)")
print(f"Frequency Domain Accuracy:         {acc_freq:.4f} ({acc_freq*100:.2f}%)")
print(f"Combined Domain Accuracy:          {acc_combined:.4f} ({acc_combined*100:.2f}%)")
print("="*70)

# Best performing
best_acc = max(acc_time, acc_freq, acc_combined)
if best_acc == acc_time:
    best_name = "Time Domain"
elif best_acc == acc_freq:
    best_name = "Frequency Domain"
else:
    best_name = "Combined"

print(f"\n🏆 BEST PERFORMER: {best_name} ({best_acc*100:.2f}%)\n")

# Comparison summary
print("📈 COMPARISON:")
print("-"*70)
if acc_time > acc_freq:
    diff = (acc_time - acc_freq) * 100
    print(f"  Time Domain is {diff:.2f}% BETTER than Frequency Domain")
else:
    diff = (acc_freq - acc_time) * 100
    print(f"  Frequency Domain is {diff:.2f}% BETTER than Time Domain")

if acc_combined > max(acc_time, acc_freq):
    print(f"  ✓ Combined approach improves performance!")
else:
    print(f"  ✗ Combined approach doesn't improve over best single domain")

# Save results folder
os.makedirs("../results", exist_ok=True)

# Save detailed results text
with open("../results/results_medium.txt", "w") as f:
    f.write("="*70 + "\n")
    f.write("PARKINSON'S DISEASE DETECTION - MEDIUM DATASET RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset Configuration:\n")
    f.write(f"  - Files per class: {MAX_FILES_PER_CLASS}\n")
    f.write(f"  - Total samples: {len(y)}\n")
    f.write(f"  - Healthy samples: {np.sum(y == 0)}\n")
    f.write(f"  - Parkinson's samples: {np.sum(y == 1)}\n\n")
    f.write(f"Feature Configuration:\n")
    f.write(f"  - Time domain features: {X_time.shape[1]}\n")
    f.write(f"  - Frequency domain features: {X_freq.shape[1]}\n")
    f.write(f"  - Combined features: {X_combined.shape[1]}\n\n")
    f.write(f"RESULTS:\n")
    f.write(f"  Time Domain Accuracy:        {acc_time:.4f} ({acc_time*100:.2f}%)\n")
    f.write(f"  Frequency Domain Accuracy:   {acc_freq:.4f} ({acc_freq*100:.2f}%)\n")
    f.write(f"  Combined Domain Accuracy:    {acc_combined:.4f} ({acc_combined*100:.2f}%)\n\n")
    f.write(f"BEST PERFORMER: {best_name} ({best_acc*100:.2f}%)\n")

# Plot comparison
labels = ['Time Domain', 'Frequency Domain', 'Combined']
accuracies = [acc_time, acc_freq, acc_combined]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

plt.figure(figsize=(12, 7))
bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85)

plt.title("Accuracy Comparison: Parkinson's Disease Detection\n(Medium Dataset - 50 Samples)", 
          fontsize=15, fontweight='bold', pad=20)
plt.ylabel("Accuracy", fontsize=13, fontweight='bold')
plt.xlabel("Feature Domain", fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2%}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add dataset info
info_text = f'Dataset: {len(y)} samples ({MAX_FILES_PER_CLASS} per class)'
plt.text(0.5, 0.02, info_text,
         transform=plt.gca().transAxes, ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.8))

plt.tight_layout()
plt.savefig("../results/accuracy_comparison_medium.png", dpi=300, bbox_inches='tight')
print("✓ Accuracy comparison plot saved!")
plt.close()

# Create detailed comparison chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

domain_names = ['Time Domain', 'Frequency Domain', 'Combined']
accuracies_list = [acc_time, acc_freq, acc_combined]
colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (ax, domain, acc, color) in enumerate(zip(axes, domain_names, accuracies_list, colors_list)):
    # Draw pie/gauge chart
    ax.barh([0], [acc], height=0.3, color=color, edgecolor='black', linewidth=2)
    ax.barh([0], [1-acc], left=[acc], height=0.3, color='lightgray', edgecolor='black', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.set_title(f'{domain}\n{acc*100:.2f}%', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Detailed Accuracy Breakdown', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../results/accuracy_breakdown_medium.png", dpi=300, bbox_inches='tight')
print("✓ Accuracy breakdown chart saved!")
plt.close()

# Print execution time
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "="*70)
print(f"⏱️  Total execution time: {minutes}m {seconds}s")
print("="*70)
print("✅ Process completed successfully!\n")