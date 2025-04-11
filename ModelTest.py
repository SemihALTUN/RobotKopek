import librosa
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# MFCC çıkarım fonksiyonu
def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Modeli yüklerken, custom_objects içinde 'mse' olarak MeanSquaredError() nesnesini veriyoruz.
model = load_model("voice_autoencoder.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Scaler'ı yükleme
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Test ses dosyasını işleme
test_file = "benim_sesim.wav"  # Test için kullanmak istediğiniz ses dosyası
features = extract_features(test_file, n_mfcc=40)
features = np.expand_dims(features, axis=0)
features_scaled = scaler.transform(features)

# Model ile yeniden oluşturma (reconstruction) işlemi
reconstructed = model.predict(features_scaled)
error = np.mean(np.square(features_scaled - reconstructed))
print("Reconstruction error:", error)

# Önceden eğitim sırasında belirlediğiniz eşik değeri (threshold)
threshold = 0.5  # Bu değeri eğitim sırasında hesapladığınız threshold ile değiştirin

if error < threshold:
    print("Test sesi kabul edildi: reconstruction error eşik değerinin altında.")
else:
    print("Test sesi reddedildi: reconstruction error eşik değerinin üzerinde.")
