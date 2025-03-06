import librosa
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

def extract_features(file_path, n_mfcc=40):
    """
    Verilen ses dosyasından 40 adet MFCC çıkarır ve zaman ortalamasını alır.
    """
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Modeli, scaler'ı ve label_map'i yükleyelim.
model = load_model("voice_classification_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Label_map'in tersini oluşturuyoruz: indeks -> sınıf adı.
inv_label_map = {v: k for k, v in label_map.items()}

# Test ses dosyasının yolunu belirleyelim.
test_file = "C:\\Users\\semii\\OneDrive\\Masaüstü\\RobotKöpek\\Test_Data\\Kisi3\\HüseyinSes.wav"

# MFCC özelliklerini çıkaralım ve normalize edelim.
features = extract_features(test_file, n_mfcc=40)
features = np.expand_dims(features, axis=0)
features_scaled = scaler.transform(features)

# Model ile tahmin yapalım.
predictions = model.predict(features_scaled)
predicted_index = np.argmax(predictions, axis=1)[0]
predicted_class = inv_label_map[predicted_index]
confidence = np.max(predictions)

print("Tahmin Edilen Sınıf:", predicted_class)
print("Güven:", confidence)
