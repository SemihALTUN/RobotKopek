import os
import numpy as np
import pickle
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa.display

# 1. Özellik Çıkarma Fonksiyonu (MFCC)
def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# 2. Verisetinin Hazırlanması
dataset_dir = "C:\\Users\\semii\\OneDrive\\Masaüstü\\RobotKöpek\\Data"  # Artırılmış ses dosyalarının bulunduğu dizin
features = []

for file_name in os.listdir(dataset_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(dataset_dir, file_name)
        feat = extract_features(file_path, n_mfcc=40)
        features.append(feat)

features = np.array(features)
print("Extracted features shape:", features.shape)

# 3. Verilerin Normalizasyonu
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Custom Metric: Reconstruction Accuracy
def reconstruction_accuracy(threshold=0.5):
    """
    Her örnek için yeniden oluşturma hatasını (MSE) hesaplar ve
    hata threshold'un altında ise 1, üstünde ise 0 alarak ortalama accuracy hesaplar.
    """
    def acc(y_true, y_pred):
        error = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return tf.reduce_mean(tf.cast(error < threshold, tf.float32))
    acc.__name__ = "reconstruction_accuracy"  # İsim ataması
    return acc

# 5. Autoencoder Modelinin Tanımlanması
input_dim = features_scaled.shape[1]  # Örneğin 40 (n_mfcc sayısı)
input_layer = layers.Input(shape=(input_dim,))
# Encoder kısmı
encoded = layers.Dense(32, activation='relu')(input_layer)
encoded = layers.Dense(16, activation='relu')(encoded)
# Decoder kısmı
decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
# Modeli compile ederken custom metric'i ekliyoruz.
autoencoder.compile(optimizer='adam', loss='mse', metrics=[reconstruction_accuracy(0.5)])
autoencoder.summary()

# 6. Model Eğitimi
history = autoencoder.fit(features_scaled, features_scaled,
                          epochs=100,
                          batch_size=8,
                          validation_split=0.2)

# 7. Reconstruction Error Hesaplama ve Threshold Belirleme
reconstructed = autoencoder.predict(features_scaled)
errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
threshold_value = np.mean(errors) + 2 * np.std(errors)
print("Belirlenen reconstruction error threshold:", threshold_value)

# Threshold'u bir dosyaya kaydedelim:
with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold_value, f)

# 8. Model ve Scaler'ın Kaydedilmesi
autoencoder.save("voice_autoencoder.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model ve scaler başarıyla kaydedildi.")

# 9. Eğitim Grafikleri: Loss ve Accuracy'nin PNG Olarak Kaydedilmesi

# Loss grafiği
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig('loss_graph.png')
plt.close()

# Reconstruction Accuracy grafiği
plt.figure()
plt.plot(history.history['reconstruction_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_reconstruction_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Reconstruction Accuracy')
plt.legend()
plt.savefig('accuracy_graph.png')
plt.close()

print("Loss ve Accuracy grafikleri kaydedildi.")

# 10. MFCC grafiklerini kaydetmek için 'mfcc' adlı klasör oluştur ve kaydet.
mfcc_dir = "mfcc"
if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

for file_name in os.listdir(dataset_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(dataset_dir, file_name)
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title("MFCC - " + file_name)
        plt.tight_layout()
        output_path = os.path.join(mfcc_dir, file_name.replace('.wav', '.png'))
        plt.savefig(output_path)
        plt.close()

print("MFCC görselleri '{}' klasörüne kaydedildi.".format(mfcc_dir))
