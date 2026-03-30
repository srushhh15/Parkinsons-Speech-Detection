import librosa

def load_audio(file_path):
    # Load first 2 seconds (faster + consistent)
    signal, sr = librosa.load(file_path, sr=22050, duration=2.0)
    return signal, sr