import numpy as np
import librosa
from scipy.stats import skew, kurtosis

# Time Domain Features (FAST - OPTIMIZED)
def extract_time_features(signal, sr):
    """Extract time domain features - FAST VERSION (no approximate entropy)"""
    
    # 1. Short-Time Energy (STE) - FAST
    frame_length = int(0.02 * sr)  # 20ms frames
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_length//2)
    ste = np.mean(np.sum(frames**2, axis=0))
    
    # 2. Zero Crossing Rate (ZCR) - FAST
    zcr_frames = librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=frame_length//2)
    zcr_mean = np.mean(zcr_frames)
    zcr_std = np.std(zcr_frames)
    
    # 3. Pitch (F0) - FAST (reduced computation)
    # Using librosa's fast pitch estimation with lower resolution
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr, fmin=50, fmax=500, threshold=0.1)
    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_contour.append(pitch)
    pitch_mean = np.mean(pitch_contour) if pitch_contour else 0
    pitch_std = np.std(pitch_contour) if pitch_contour else 0
    
    # 4. Jitter - FAST
    if len(pitch_contour) > 1:
        jitter = np.mean(np.abs(np.diff(pitch_contour))) / np.mean(pitch_contour) if np.mean(pitch_contour) > 0 else 0
    else:
        jitter = 0
    
    # 5. Shimmer - FAST
    shimmer = np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
    
    # 6. Energy Entropy - FAST
    energy = np.abs(signal)**2
    energy_norm = energy / np.sum(energy)
    energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-10))
    
    # 7. Signal Skewness - FAST
    signal_skew = skew(signal)
    
    # 8. Signal Kurtosis - FAST
    signal_kurtosis = kurtosis(signal)
    
    # 9. RMS Energy - FAST (simple & effective)
    rms_energy = np.sqrt(np.mean(signal**2))
    
    return [
        ste,              # Short-Time Energy
        zcr_mean,         # Zero Crossing Rate (mean)
        zcr_std,          # ZCR std dev
        pitch_mean,       # Pitch (mean)
        pitch_std,        # Pitch std dev
        jitter,           # Jitter
        shimmer,          # Shimmer
        energy_entropy,   # Energy Entropy
        signal_skew,      # Skewness
        signal_kurtosis,  # Kurtosis
        rms_energy        # RMS Energy
    ]


# Frequency Domain Features (FINAL)
def extract_freq_features(signal, sr):
    """Extract frequency domain features - FAST"""
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Spectral Features
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))

    # Chroma Feature
    chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))

    return list(mfcc_mean) + list(delta_mean) + [centroid, bandwidth, rolloff, chroma]