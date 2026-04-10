import numpy as np
import librosa
from scipy.stats import skew, kurtosis

# ============= VOWEL FORMANT EXTRACTION =============
def extract_vowel_formants(signal, sr):
    """Extract vowel formant features - ONLY for vowel datasets
    
    Returns: [F1_mean, F1_std, F2_mean, F2_std, F3_mean, F3_std]
    """
    try:
        # LPC order for vowel formants
        order = 12
        
        # Frame the signal for vowel analysis
        frame_length = int(0.03 * sr)  # 30ms frames (optimal for vowels)
        hop_length = frame_length // 2
        frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
        
        all_formants = []
        
        for frame in frames.T:
            # Apply Hamming window
            frame = frame * np.hamming(len(frame))
            
            # Autocorrelation for LPC
            r = np.correlate(frame, frame, mode='full')
            r = r[len(r)//2:]
            r[0] = r[0] * (1 + 1e-5)
            
            # Levinson-Durbin recursion for LPC coefficients
            A = np.zeros(order + 1)
            A[0] = 1.0
            E = r[0]
            
            for i in range(1, order + 1):
                if E > 0:
                    alpha = -np.sum(A[1:i] * r[i:0:-1]) / E
                else:
                    alpha = 0
                A_new = np.zeros(order + 1)
                A_new[0] = 1.0
                A_new[1:i+1] = A[1:i] + alpha * A[i-1:0:-1]
                A = A_new
                E *= (1 - alpha * alpha) if (1 - alpha * alpha) > 0 else 1e-10
            
            # Extract formant frequencies from LPC roots
            if np.max(np.abs(A[1:])) > 0:
                roots = np.roots(A)
                formant_freqs = []
                for root in roots:
                    if np.abs(root) < 1:
                        freq = sr * np.angle(root) / (2 * np.pi)
                        if 50 < freq < 5000:  # Valid speech frequency range
                            formant_freqs.append(freq)
                
                formant_freqs = sorted(formant_freqs)[:3]  # Top 3 formants
                while len(formant_freqs) < 3:
                    formant_freqs.append(0)
                all_formants.append(formant_freqs)
        
        # Calculate statistics for each formant
        if len(all_formants) > 0:
            formants_array = np.array(all_formants)
            f1_mean = np.mean(formants_array[:, 0])
            f1_std = np.std(formants_array[:, 0])
            f2_mean = np.mean(formants_array[:, 1])
            f2_std = np.std(formants_array[:, 1])
            f3_mean = np.mean(formants_array[:, 2])
            f3_std = np.std(formants_array[:, 2])
        else:
            f1_mean = f1_std = f2_mean = f2_std = f3_mean = f3_std = 0
        
        return [f1_mean, f1_std, f2_mean, f2_std, f3_mean, f3_std]
    
    except Exception as e:
        # Fallback if formant extraction fails
        return [0, 0, 0, 0, 0, 0]

# ========================================================

# Time Domain Features (FAST - OPTIMIZED)
def extract_time_features(signal, sr):
    """Extract time domain features - FAST VERSION"""
    
    # 1. Short-Time Energy (STE)
    frame_length = int(0.02 * sr)  # 20ms frames
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_length//2)
    ste = np.mean(np.sum(frames**2, axis=0))
    
    # 2. Zero Crossing Rate (ZCR)
    zcr_frames = librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=frame_length//2)
    zcr_mean = np.mean(zcr_frames)
    zcr_std = np.std(zcr_frames)
    
    # 3. Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr, fmin=50, fmax=500, threshold=0.1)
    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_contour.append(pitch)
    pitch_mean = np.mean(pitch_contour) if pitch_contour else 0
    pitch_std = np.std(pitch_contour) if pitch_contour else 0
    
    # 4. Jitter
    if len(pitch_contour) > 1:
        jitter = np.mean(np.abs(np.diff(pitch_contour))) / np.mean(pitch_contour) if np.mean(pitch_contour) > 0 else 0
    else:
        jitter = 0
    
    # 5. Shimmer
    shimmer = np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
    
    # 6. Energy Entropy
    energy = np.abs(signal)**2
    energy_norm = energy / np.sum(energy)
    energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-10))
    
    # 7. Signal Skewness
    signal_skew = skew(signal)
    
    # 8. Signal Kurtosis
    signal_kurtosis = kurtosis(signal)
    
    # 9. RMS Energy
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


# Frequency Domain Features
def extract_freq_features(signal, sr):
    """Extract frequency domain features"""
    
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