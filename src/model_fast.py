from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os

def train_model_fast(X, y, domain_name="Model"):
    """Fast training with detailed metrics for medium-sized datasets"""
    
    print(f"\nTraining {domain_name}...")
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Split data (70/30 for more reliable validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train SVM
    print(f"   Training SVM model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n   RESULTS:")
    print(f"   - Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"   - Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"   - Recall:      {recall:.4f} ({recall*100:.2f}%) [Sensitivity]")
    print(f"   - Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   - F1-Score:    {f1:.4f}")
    
    print(f"\n   CONFUSION MATRIX:")
    print(f"   - True Negatives:  {tn} (Healthy correctly identified)")
    print(f"   - False Positives: {fp} (Healthy misclassified as PD)")
    print(f"   - False Negatives: {fn} (PD missed)")
    print(f"   - True Positives:  {tp} (PD correctly identified)")
    
    # Plot confusion matrix
    os.makedirs("../results", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', "PD"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {domain_name}\n(Accuracy: {acc*100:.2f}%)", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filename = domain_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    plt.savefig(f"../results/cm_{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return acc