import numpy as np
import librosa

# Time Domain Features
def extract_time_features(signal, sr):
    ste = np.sum(signal**2) / len(signal)

    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    pitches, _ = librosa.piptrack(y=signal, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    jitter = np.std(signal)
    shimmer = np.mean(np.abs(signal))

    return [ste, zcr, pitch, jitter, shimmer]


# Frequency Domain Features (FINAL)
def extract_freq_features(signal, sr):
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

    # Chroma Feature (NEW)
    chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))

    return list(mfcc_mean) + list(delta_mean) + [centroid, bandwidth, rolloff, chroma]