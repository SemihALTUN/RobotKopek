import os
import numpy as np
import pickle
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Eğitim sırasında belirlediğiniz threshold değeri (bu değeri model eğitiminizden almanız gerekir)
threshold = 0.5

def extract_features(file_path, n_mfcc=40):
    """
    Verilen ses dosyasından 40 adet MFCC çıkarır ve zaman ortalaması alır.
    """
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Modeli, custom loss 'mse' objesini tanıtarak yüklüyoruz.
model = load_model("voice_autoencoder.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Scaler'ı yükleme
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Test veri dizini yapısı:
# test_data/my_voice : modelin kabul etmesi gereken (doğru/pozitif) ses örnekleri. Label = 1
# test_data/others  : modelin reddetmesi gereken (yanlış/negatif) ses örnekleri. Label = 0
# Not: Eğer "others" klasöründe de yalnızca sizin sesiniz varsa, raporlama tek sınıflı olarak gerçekleşecektir.
test_dirs = {"BenimSesim": 1, "DiğerSesler": 0}

y_true = []
y_pred = []

for folder, label in test_dirs.items():
    folder_path = os.path.join("Test_Data", folder)
    if not os.path.exists(folder_path):
        print(f"Test klasörü '{folder}' mevcut değil. Atlanıyor.")
        continue
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path, n_mfcc=40)
            features = np.expand_dims(features, axis=0)
            features_scaled = scaler.transform(features)
            reconstructed = model.predict(features_scaled)
            error = np.mean(np.square(features_scaled - reconstructed))
            # Eğer hata eşik değerinin altındaysa 'benim sesim' olarak kabul et (1), aksi halde 0
            prediction = 1 if error < threshold else 0
            y_true.append(label)
            y_pred.append(prediction)
            print(f"Dosya: {file_name}, Reconstruction Error: {error:.4f}, Tahmin: {prediction}")

if len(y_true) > 0:
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["others", "my_voice"])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(report)
else:
    print("Test dosyası bulunamadı.")

# Model hakkında özet bilgi
model.summary()
'''1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step
Dosya: benim_sesim.wav, Reconstruction Error: 0.4934, Tahmin: 1
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
Dosya: benim_sesim2.wav, Reconstruction Error: 0.7061, Tahmin: 0
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
Dosya: AnnemDeneme.wav, Reconstruction Error: 0.6257, Tahmin: 0
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
Dosya: BabamDeneme.wav, Reconstruction Error: 0.6130, Tahmin: 0
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
Dosya: ZeynepDeneme.wav, Reconstruction Error: 1.0506, Tahmin: 0

Confusion Matrix:
[[3 0]
 [1 1]]

Accuracy: 0.8

Classification Report:
              precision    recall  f1-score   support

      others       0.75      1.00      0.86         3
    my_voice       1.00      0.50      0.67         2

    accuracy                           0.80         5
   macro avg       0.88      0.75      0.76         5
weighted avg       0.85      0.80      0.78         5

Model: "functional"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ input_layer (InputLayer)        │ (None, 40)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │         1,312 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 32)             │           544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 40)             │         1,320 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,706 (14.48 KB)
 Trainable params: 3,704 (14.47 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)'''