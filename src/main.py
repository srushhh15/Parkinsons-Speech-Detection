import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_audio
from features import extract_time_features, extract_freq_features
from model import train_model

DATA_PATH = "../data"

X_time = []
X_freq = []
y = []

MAX_FILES = 50  # increase or remove for full dataset

# Load Data
for label in ["healthy", "parkinson"]:
    folder = os.path.join(DATA_PATH, label)

    files = os.listdir(folder)[:MAX_FILES]

    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder, file)

            signal, sr = load_audio(file_path)

            time_feat = extract_time_features(signal, sr)
            freq_feat = extract_freq_features(signal, sr)

            X_time.append(time_feat)
            X_freq.append(freq_feat)

            y.append(0 if label == "healthy" else 1)

# Convert
X_time = np.array(X_time)
X_freq = np.array(X_freq)
y = np.array(y)

# Train
acc_time = train_model(X_time, y)
acc_freq = train_model(X_freq, y)

# Print
print("\n========== RESULTS ==========")
print("Time Domain Accuracy:", acc_time)
print("Frequency Domain Accuracy:", acc_freq)

# Save results folder
os.makedirs("../results", exist_ok=True)

# Save results text
with open("../results/results.txt", "w") as f:
    f.write(f"Time Domain Accuracy: {acc_time}\n")
    f.write(f"Frequency Domain Accuracy: {acc_freq}\n")

# Plot
labels = ['Time Domain', 'Frequency Domain']
accuracies = [acc_time, acc_freq]

plt.figure()
plt.bar(labels, accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Save graph
plt.savefig("../results/accuracy_plot.png")

# Show
plt.show()