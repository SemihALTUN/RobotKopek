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

# Sınıflandırma modelini yükleme
model = load_model("voice_classification_model.h5")

# Scaler ve label_map'i yükleme
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# label_map'in tersini oluşturuyoruz: indeks -> sınıf adı
inv_label_map = {v: k for k, v in label_map.items()}

# İstediğiniz isim eşlemesi: Eğer model "Kisi1" çıkarsa "enes", "Kisi2" için "Semih", "Kisi3" için "Hüseyin"
name_mapping = {
    "Kisi1": "enes",
    "Kisi2": "Semih",
    "Kisi3": "Hüseyin"
}

# Mikrofon ile ses kaydı alıyoruz
duration = 5  # Kaydın süresi (saniye)
audio, fs = record_audio(duration=duration, fs=16000)

# Kaydedilen sesten MFCC özelliklerini çıkarıyoruz
features = extract_features_from_audio(audio, sr=fs, n_mfcc=40)
features = np.expand_dims(features, axis=0)
features_scaled = scaler.transform(features)

# Model ile tahmin yapma
predictions = model.predict(features_scaled)
predicted_index = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions)

# Modelin tahmin ettiği sınıf adını alıp, isim eşlemesini uyguluyoruz
predicted_class = inv_label_map[predicted_index]
mapped_name = name_mapping.get(predicted_class, predicted_class)

print("Tahmin Edilen Sınıf:", predicted_class)
print("Eşleştirilmiş İsim:", mapped_name)
print("Güven:", confidence)
