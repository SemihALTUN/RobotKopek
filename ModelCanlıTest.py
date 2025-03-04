import sounddevice as sd
import librosa
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Mikrofon kaydı için fonksiyon (örneğin 5 saniyelik kayıt)
def record_audio(duration=5, fs=16000):
    print("Lütfen konuşmaya başlayın...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Kayıt tamamlanana kadar bekler
    audio = np.squeeze(audio)
    return audio, fs

# Kaydedilen ses sinyalinden MFCC özelliklerini çıkaran fonksiyon
def extract_features_from_audio(audio, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Modeli, custom_objects içinde 'mse' olarak MeanSquaredError() nesnesini vererek yüklüyoruz.
model = load_model("voice_autoencoder.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Scaler'ı yükleme
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Eğitim sırasında kaydedilmiş threshold'u yükleyelim
with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

# Mikrofon ile ses kaydı alıyoruz
duration = 5  # Kaydın süresi (saniye)
audio, fs = record_audio(duration=duration, fs=16000)

# Kaydedilen sesten MFCC özelliklerini çıkarıyoruz
features = extract_features_from_audio(audio, sr=fs, n_mfcc=40)
features = np.expand_dims(features, axis=0)
features_scaled = scaler.transform(features)

# Model ile yeniden oluşturma (reconstruction) işlemi
reconstructed = model.predict(features_scaled)
error = np.mean(np.square(features_scaled - reconstructed))
print("Reconstruction error:", error)

# Hesaplanan threshold ile karşılaştırma
if error < threshold:
    print("Test sesi kabul edildi: reconstruction error eşik değerinin altında.")
else:
    print("Test sesi reddedildi: reconstruction error eşik değerinin üzerinde.")
