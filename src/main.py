import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import traceback

# Set absolute path for data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

print(f"DEBUG: Script directory: {SCRIPT_DIR}")
print(f"DEBUG: Project root: {PROJECT_ROOT}")
print(f"DEBUG: Data path: {DATA_PATH}")
print(f"DEBUG: Data path exists: {os.path.exists(DATA_PATH)}\n")

from preprocessing import load_audio
from features_fast import extract_time_features, extract_freq_features, extract_vowel_formants
from model_fast import train_model_fast

# ============ CONFIGURATION ============
DATASET_TYPES = ['A', 'E', 'I', 'O', 'U', 'apto', 'petaka']
VOWELS = ['A', 'E', 'I', 'O', 'U']
MAX_FILES_PER_CLASS = 25
# =======================================

start_time = time.time()
all_results = {}

print("\n" + "="*80)
print("UNIFIED PARKINSON'S DISEASE DETECTION - VOWELS & LANGUAGE DATASETS")
print("="*80)
print(f"Processing {len(DATASET_TYPES)} datasets:")
print(f"  Vowels: {', '.join(VOWELS)}")
print(f"  Language: apto, petaka")
print(f"Features: Time Domain (11) + Frequency Domain (17) + Vowel Formants (6 for vowels only)")
print(f"Total: 34 features (Vowels) | 28 features (Language)\n")

# ============ MAIN PROCESSING LOOP ============
for dataset_type in DATASET_TYPES:
    try:
        print("\n" + "="*80)
        print(f"PROCESSING DATASET: {dataset_type.upper()}")
        print("="*80)
        
        X_time = []
        X_freq = []
        X_formants = []
        X_combined = []
        y = []
        
        dataset_path = os.path.join(DATA_PATH, dataset_type)
        
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset folder not found: {dataset_path}")
            continue
        
        # ---- LOAD HEALTHY SAMPLES ----
        healthy_folder = os.path.join(dataset_path, "healthy")
        if os.path.exists(healthy_folder):
            files = [f for f in os.listdir(healthy_folder) if f.lower().endswith(".wav")][:MAX_FILES_PER_CLASS]
            print(f"\nLoading {len(files)} HEALTHY samples...")
            
            loaded_count = 0
            for idx, file in enumerate(files, 1):
                try:
                    signal, sr = load_audio(os.path.join(healthy_folder, file))
                    
                    time_feat = extract_time_features(signal, sr)
                    freq_feat = extract_freq_features(signal, sr)
                    
                    X_time.append(time_feat)
                    X_freq.append(freq_feat)
                    
                    # Add formants only for vowels
                    if dataset_type in VOWELS:
                        formant_feat = extract_vowel_formants(signal, sr)
                        X_formants.append(formant_feat)
                        combined_feat = time_feat + freq_feat + formant_feat
                    else:
                        X_formants.append([0]*6)  # Placeholder for language datasets
                        combined_feat = time_feat + freq_feat
                    
                    X_combined.append(combined_feat)
                    y.append(0)  # Healthy
                    loaded_count += 1
                    
                    # Progress bar
                    progress = int((idx / len(files)) * 20)
                    bar = "#" * progress + "-" * (20 - progress)
                    print(f"  [{bar}] {idx}/{len(files)}", end='\r')
                    
                except Exception as e:
                    print(f"  WARNING: Error loading {file}: {str(e)}")
                    continue
            
            print(f"\n  Loaded {loaded_count}/{len(files)} healthy files")
        
        # ---- LOAD PD (PARKINSON'S DISEASE) SAMPLES ----
        pd_folder = os.path.join(dataset_path, "PD")
        if os.path.exists(pd_folder):
            files = [f for f in os.listdir(pd_folder) if f.lower().endswith(".wav")][:MAX_FILES_PER_CLASS]
            print(f"\nLoading {len(files)} PD (PARKINSON'S DISEASE) samples...")
            
            loaded_count = 0
            for idx, file in enumerate(files, 1):
                try:
                    signal, sr = load_audio(os.path.join(pd_folder, file))
                    
                    time_feat = extract_time_features(signal, sr)
                    freq_feat = extract_freq_features(signal, sr)
                    
                    X_time.append(time_feat)
                    X_freq.append(freq_feat)
                    
                    # Add formants only for vowels
                    if dataset_type in VOWELS:
                        formant_feat = extract_vowel_formants(signal, sr)
                        X_formants.append(formant_feat)
                        combined_feat = time_feat + freq_feat + formant_feat
                    else:
                        X_formants.append([0]*6)  # Placeholder for language datasets
                        combined_feat = time_feat + freq_feat
                    
                    X_combined.append(combined_feat)
                    y.append(1)  # Parkinson's Disease
                    loaded_count += 1
                    
                    # Progress bar
                    progress = int((idx / len(files)) * 20)
                    bar = "#" * progress + "-" * (20 - progress)
                    print(f"  [{bar}] {idx}/{len(files)}", end='\r')
                    
                except Exception as e:
                    print(f"  WARNING: Error loading {file}: {str(e)}")
                    continue
            
            print(f"\n  Loaded {loaded_count}/{len(files)} PD files")
        
        # ---- CONVERT TO NUMPY ARRAYS ----
        X_time = np.array(X_time)
        X_freq = np.array(X_freq)
        X_formants = np.array(X_formants)
        X_combined = np.array(X_combined)
        y = np.array(y)
        
        if len(y) < 10:
            print(f"\nERROR: Not enough samples for {dataset_type}! Need at least 10 samples")
            print(f"Skipping {dataset_type}...\n")
            continue
        
        # ---- DATASET SUMMARY ----
        print("\n" + "-"*80)
        print(f"DATASET SUMMARY: {dataset_type.upper()}")
        print("-"*80)
        print(f"Total samples:        {len(y)}")
        print(f"  - Healthy:          {np.sum(y == 0)}")
        print(f"  - Parkinson's (PD): {np.sum(y == 1)}")
        print(f"  - Balance:          {np.sum(y == 0) / len(y) * 100:.1f}% vs {np.sum(y == 1) / len(y) * 100:.1f}%")
        
        if dataset_type in VOWELS:
            print(f"\nFeatures:")
            print(f"  Time Domain:        {X_time.shape[1]} features")
            print(f"  Frequency Domain:   {X_freq.shape[1]} features")
            print(f"  Vowel Formants:     {X_formants.shape[1]} features")
            print(f"  Combined:           {X_combined.shape[1]} features")
        else:
            print(f"\nFeatures (No Formants for Language Dataset):")
            print(f"  Time Domain:        {X_time.shape[1]} features")
            print(f"  Frequency Domain:   {X_freq.shape[1]} features")
            print(f"  Combined:           {X_combined.shape[1]} features")
        
        # ---- NORMALIZE FEATURES ----
        print(f"\nNORMALIZING FEATURES")
        X_time_norm = StandardScaler().fit_transform(X_time)
        X_freq_norm = StandardScaler().fit_transform(X_freq)
        X_combined_norm = StandardScaler().fit_transform(X_combined)
        
        # ---- TRAIN MODELS ----
        print(f"\nTRAINING MODELS FOR {dataset_type.upper()}")
        print("-"*80)
        
        try:
            acc_time = train_model_fast(X_time_norm, y, f"Time_Domain_{dataset_type}")
        except Exception as e:
            print(f"ERROR training Time Domain model: {str(e)}")
            acc_time = 0
        
        print()
        
        try:
            acc_freq = train_model_fast(X_freq_norm, y, f"Frequency_Domain_{dataset_type}")
        except Exception as e:
            print(f"ERROR training Frequency Domain model: {str(e)}")
            acc_freq = 0
        
        print()
        
        try:
            acc_combined = train_model_fast(X_combined_norm, y, f"Combined_{dataset_type}")
        except Exception as e:
            print(f"ERROR training Combined model: {str(e)}")
            acc_combined = 0
        
        # ---- STORE RESULTS ----
        all_results[dataset_type] = {
            'time_accuracy': acc_time,
            'freq_accuracy': acc_freq,
            'combined_accuracy': acc_combined,
            'total_samples': len(y),
            'healthy_count': np.sum(y == 0),
            'parkinson_count': np.sum(y == 1),
            'is_vowel': dataset_type in VOWELS
        }
        
        # ---- PRINT RESULTS ----
        print("\n" + "="*80)
        print(f"RESULTS FOR {dataset_type.upper()}")
        print("="*80)
        print(f"Time Domain Accuracy:              {acc_time*100:.2f}%")
        print(f"Frequency Domain Accuracy:         {acc_freq*100:.2f}%")
        print(f"Combined Accuracy:                 {acc_combined*100:.2f}%")
        print("="*80)
    
    except Exception as e:
        print(f"\nCRITICAL ERROR processing {dataset_type}:")
        print(traceback.format_exc())
        print(f"Skipping to next dataset...\n")
        continue

# ============ OVERALL COMPARISON ============
print("\n\n" + "="*80)
print("COMPREHENSIVE COMPARISON - ALL 7 DATASETS")
print("="*80)

if len(all_results) == 0:
    print("ERROR: No results generated! Check your data folders and files.")
else:
    print("\nDETAILED RESULTS BY DATASET:\n")
    
    for dataset_type in DATASET_TYPES:
        if dataset_type in all_results:
            r = all_results[dataset_type]
            dataset_label = "VOWEL" if r['is_vowel'] else "LANGUAGE"
            print(f"{dataset_label} {dataset_type.upper()}")
            print(f"  Samples: {r['total_samples']} (Healthy: {r['healthy_count']}, PD: {r['parkinson_count']})")
            print(f"  Time Domain:       {r['time_accuracy']*100:.2f}%")
            print(f"  Frequency Domain:  {r['freq_accuracy']*100:.2f}%")
            print(f"  Combined:          {r['combined_accuracy']*100:.2f}%")
            print()
    
    # ---- CALCULATE STATISTICS ----
    combined_accuracies = [r['combined_accuracy'] for r in all_results.values()]
    vowel_accuracies = [r['combined_accuracy'] for d, r in all_results.items() if r['is_vowel']]
    language_accuracies = [r['combined_accuracy'] for d, r in all_results.items() if not r['is_vowel']]
    
    avg_overall = np.mean(combined_accuracies)
    avg_vowel = np.mean(vowel_accuracies) if vowel_accuracies else 0
    avg_language = np.mean(language_accuracies) if language_accuracies else 0
    
    best_dataset = max(all_results.items(), key=lambda x: x[1]['combined_accuracy'])
    worst_dataset = min(all_results.items(), key=lambda x: x[1]['combined_accuracy'])
    
    vowel_items = [(d, r) for d, r in all_results.items() if r['is_vowel']]
    language_items = [(d, r) for d, r in all_results.items() if not r['is_vowel']]
    
    best_vowel = max(vowel_items, key=lambda x: x[1]['combined_accuracy']) if vowel_items else None
    best_language = max(language_items, key=lambda x: x[1]['combined_accuracy']) if language_items else None
    
    print("\n" + "-"*80)
    print("STATISTICAL ANALYSIS")
    print("-"*80)
    print(f"Average Overall Accuracy (All {len(all_results)} datasets):  {avg_overall*100:.2f}%")
    if vowel_accuracies:
        print(f"Average Vowel Accuracy ({len(vowel_accuracies)} vowels):          {avg_vowel*100:.2f}%")
    if language_accuracies:
        print(f"Average Language Accuracy ({len(language_accuracies)} datasets):     {avg_language*100:.2f}%")
    print()
    print(f"BEST OVERALL DATASET:      {best_dataset[0].upper()} ({best_dataset[1]['combined_accuracy']*100:.2f}%)")
    print(f"MOST CHALLENGING DATASET:  {worst_dataset[0].upper()} ({worst_dataset[1]['combined_accuracy']*100:.2f}%)")
    if best_vowel:
        print(f"BEST VOWEL DATASET:        {best_vowel[0].upper()} ({best_vowel[1]['combined_accuracy']*100:.2f}%)")
    if best_language:
        print(f"BEST LANGUAGE DATASET:     {best_language[0].upper()} ({best_language[1]['combined_accuracy']*100:.2f}%)")
    
    # ============ SAVE RESULTS ============
    os.makedirs("../results", exist_ok=True)
    
    # Save text results
    with open("../results/results_all_datasets.txt", "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("UNIFIED PARKINSON'S DISEASE DETECTION - ALL 7 DATASETS\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASETS ANALYZED:\n")
        f.write("  VOWELS: A, E, I, O, U (with formant extraction)\n")
        f.write("  LANGUAGE: apto, petaka (without formants)\n\n")
        
        f.write("FEATURES EXTRACTED:\n")
        f.write("  TIME DOMAIN: STE, ZCR, Pitch, Jitter, Shimmer, Energy Entropy, Skewness, Kurtosis, RMS Energy (11 features)\n")
        f.write("  FREQUENCY DOMAIN: MFCC, Delta MFCC, Spectral Centroid, Bandwidth, Rolloff, Chroma (17 features)\n")
        f.write("  VOWEL FORMANTS: F1, F2, F3 (mean and std) (6 features - vowels only)\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS BY DATASET\n")
        f.write("="*80 + "\n\n")
        
        for dataset_type in DATASET_TYPES:
            if dataset_type in all_results:
                r = all_results[dataset_type]
                dataset_label = "VOWEL" if r['is_vowel'] else "LANGUAGE"
                f.write(f"{dataset_label}: {dataset_type.upper()}\n")
                f.write(f"{'─'*80}\n")
                f.write(f"Total Samples:                {r['total_samples']}\n")
                f.write(f"  - Healthy:                  {r['healthy_count']}\n")
                f.write(f"  - Parkinson's Disease (PD): {r['parkinson_count']}\n")
                f.write(f"  - Balance:                  {r['healthy_count']/r['total_samples']*100:.1f}% vs {r['parkinson_count']/r['total_samples']*100:.1f}%\n\n")
                f.write(f"Accuracy Results:\n")
                f.write(f"  Time Domain:                {r['time_accuracy']*100:.2f}%\n")
                f.write(f"  Frequency Domain:           {r['freq_accuracy']*100:.2f}%\n")
                f.write(f"  Combined:                   {r['combined_accuracy']*100:.2f}%\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average Overall Accuracy (All {len(all_results)}):   {avg_overall*100:.2f}%\n")
        if vowel_accuracies:
            f.write(f"Average Vowel Accuracy ({len(vowel_accuracies)}):         {avg_vowel*100:.2f}%\n")
        if language_accuracies:
            f.write(f"Average Language Accuracy ({len(language_accuracies)}):      {avg_language*100:.2f}%\n\n")
        f.write(f"Best Overall: {best_dataset[0].upper()} ({best_dataset[1]['combined_accuracy']*100:.2f}%)\n")
        f.write(f"Most Challenging: {worst_dataset[0].upper()} ({worst_dataset[1]['combined_accuracy']*100:.2f}%)\n")
        if best_vowel:
            f.write(f"Best Vowel: {best_vowel[0].upper()} ({best_vowel[1]['combined_accuracy']*100:.2f}%)\n")
        if best_language:
            f.write(f"Best Language: {best_language[0].upper()} ({best_language[1]['combined_accuracy']*100:.2f}%)\n")
    
    print(f"\nResults saved: ../results/results_all_datasets.txt")
    
    # ============ CREATE VISUALIZATIONS ============
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        dataset_names = list(all_results.keys())
        time_accs = [all_results[d]['time_accuracy'] * 100 for d in dataset_names]
        freq_accs = [all_results[d]['freq_accuracy'] * 100 for d in dataset_names]
        comb_accs = [all_results[d]['combined_accuracy'] * 100 for d in dataset_names]
        
        # Plot 1: Time Domain
        ax = axes[0, 0]
        colors = ['#FF6B6B' if all_results[d]['is_vowel'] else '#FFB347' for d in dataset_names]
        bars = ax.bar(dataset_names, time_accs, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_title('Time Domain Accuracy', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, time_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{acc:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: Frequency Domain
        ax = axes[0, 1]
        bars = ax.bar(dataset_names, freq_accs, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_title('Frequency Domain Accuracy', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, freq_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{acc:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 3: Combined
        ax = axes[0, 2]
        bars = ax.bar(dataset_names, comb_accs, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_title('Combined Accuracy', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, comb_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{acc:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 4: Comparison across domains
        ax = axes[1, 0]
        x = np.arange(len(dataset_names))
        width = 0.25
        ax.bar(x - width, time_accs, width, label='Time Domain', color='#FF6B6B', edgecolor='black', linewidth=1.5)
        ax.bar(x, freq_accs, width, label='Frequency Domain', color='#4ECDC4', edgecolor='black', linewidth=1.5)
        ax.bar(x + width, comb_accs, width, label='Combined', color='#45B7D1', edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Comparison Across All Domains', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Plot 5: Vowels vs Language
        ax = axes[1, 1]
        vowel_data = [all_results[d]['combined_accuracy'] * 100 for d in VOWELS if d in all_results]
        lang_data = [all_results[d]['combined_accuracy'] * 100 for d in ['apto', 'petaka'] if d in all_results]
        
        if vowel_data or lang_data:
            bp = ax.boxplot([vowel_data, lang_data], tick_labels=['Vowels', 'Language'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#FF9999', '#FFD9B3']):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)
        
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Vowel vs Language Performance', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Plot 6: Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""SUMMARY STATISTICS

Overall Average:     {avg_overall:.2f}%
Vowel Average:       {avg_vowel:.2f}%
Language Average:    {avg_language:.2f}%

Best Dataset:
  {best_dataset[0].upper()} - {best_dataset[1]['combined_accuracy']*100:.2f}%

Most Challenging:
  {worst_dataset[0].upper()} - {worst_dataset[1]['combined_accuracy']*100:.2f}%

Best Vowel:
  {best_vowel[0].upper() if best_vowel else 'N/A'}

Best Language:
  {best_language[0].upper() if best_language else 'N/A'}

Total Datasets: {len(all_results)}
  Vowels: {len(vowel_accuracies)}
  Language: {len(language_accuracies)}

Total Samples: {sum([r['total_samples'] for r in all_results.values()])}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', 
                alpha=0.8, pad=1, edgecolor='black', linewidth=2))
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Vowels'),
            Patch(facecolor='#FFB347', edgecolor='black', label='Language')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  ncol=2, fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle('Unified Parkinson Disease Detection - Comprehensive Analysis\nAll Datasets (Vowels + Language-Based)',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("../results/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        print("Comprehensive analysis plot saved: ../results/comprehensive_analysis.png")
        plt.close()
    
    except Exception as e:
        print(f"ERROR creating visualizations: {str(e)}")
        print(traceback.format_exc())

# ============ EXECUTION TIME ============
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "="*80)
print(f"Total execution time: {minutes}m {seconds}s")
print("="*80)
print("\nUNIFIED ANALYSIS COMPLETED SUCCESSFULLY!\n")
print("Generated files:")
print("  - results/results_all_datasets.txt")
print("  - results/comprehensive_analysis.png")
print("  - results/cm_*.png (Confusion Matrices for each dataset)\n")