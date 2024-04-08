import librosa
import librosa.display

# Load audio file
audio_file = "your_audio_file.wav"
y, sr = librosa.load(audio_file)
hop_length = 512

# Extract Chroma features (7 x n)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Extract Spectral Contrast features (6 x n)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Extract Tonnetz features (12 x n)
tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

# Combine features cst
cst = np.concatenate((chroma, spectral_contrast, tonnetz), axis=0)

# Transpose to get the desired shape (n x m)
combined_features = cst.T


# Define coefficients for linear superposition
coefficients = {
    "mfcc": 0.3,
    "log_mel_spec": 0.4,
    "cst": 0.3
}

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Extract Log-Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# Calculate the combined feature M-LM-C using linear superposition
m_lm_c = (
    coefficients["mfcc"] * mfccs +
    coefficients["log_mel_spec"] * log_mel_spec +
    coefficients["cst"] * cst
)
