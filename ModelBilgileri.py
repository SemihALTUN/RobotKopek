import os
import numpy as np
import pickle
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def extract_features(file_path, n_mfcc=40):
    """
    Verilen ses dosyasından 40 adet MFCC çıkarır ve zaman ortalamasını alır.
    """
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Sınıflandırma modelini yükleyelim.
model = load_model("voice_classification_model.h5")

# Scaler ve label_map dosyalarını yükleyelim.
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# label_map'in tersini oluşturuyoruz: indeks -> sınıf adı
inv_label_map = {v: k for k, v in label_map.items()}

# Test veri dizini: Test_Data altında her kişinin seslerini içeren klasörler mevcut.
test_data_dir = "Test_Data"

y_true = []
y_pred = []

# Test_Data klasöründeki tüm alt klasörleri dolaşıyoruz.
for folder in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, folder)
    # Sadece dizinleri alalım
    if not os.path.isdir(folder_path):
        continue
    # Klasör ismini normalize edelim: örneğin "kisi1" -> "Kisi1"
    folder_norm = folder.capitalize()  # ya da folder.lower().capitalize()
    if folder_norm not in label_map:
        print(f"Test klasörü '{folder}' label_map'te bulunamadı. Atlanıyor.")
        continue
    label = label_map[folder_norm]
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path, n_mfcc=40)
            features = np.expand_dims(features, axis=0)
            features_scaled = scaler.transform(features)
            predictions = model.predict(features_scaled)
            predicted_label = np.argmax(predictions, axis=1)[0]
            y_true.append(label)
            y_pred.append(predicted_label)
            print(f"Dosya: {file_name}, Tahmin: {inv_label_map[predicted_label]} (Label: {predicted_label})")

if len(y_true) > 0:
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=[inv_label_map[i] for i in sorted(inv_label_map.keys())])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(report)
else:
    print("Test dosyası bulunamadı.")

model.summary()
'''
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
Dosya: EnesSes.wav, Tahmin: Kisi1 (Label: 0)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
Dosya: EnesSes2.wav, Tahmin: Kisi1 (Label: 0)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
Dosya: SemihSes.wav, Tahmin: Kisi2 (Label: 1)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
Dosya: SemihSes2.wav, Tahmin: Kisi2 (Label: 1)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
Dosya: HüseyinSes.wav, Tahmin: Kisi3 (Label: 2)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
Dosya: HüseyinSes2.wav, Tahmin: Kisi3 (Label: 2)

Confusion Matrix:
[[2 0 0]
 [0 2 0]
 [0 0 2]]

Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

       Kisi1       1.00      1.00      1.00         2
       Kisi2       1.00      1.00      1.00         2
       Kisi3       1.00      1.00      1.00         2

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6

Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │         1,312 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 3)              │            51 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,893 (7.40 KB)
 Trainable params: 1,891 (7.39 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)

Process finished with exit code 0

'''