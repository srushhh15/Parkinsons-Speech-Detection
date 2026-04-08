import librosa
import numpy as np

def load_audio(file_path, sr=22050):
    """Load audio file and return signal and sample rate"""
    signal, sr = librosa.load(file_path, sr=sr)
    return signal, sr