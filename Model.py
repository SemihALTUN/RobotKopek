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

# 2. Verisetinin Hazırlanması (Her kişinin sesleri ayrı klasörde)
dataset_dir = "C:\\Users\\semii\\OneDrive\\Masaüstü\\RobotKöpek\\Data"
features = []
labels = []
# Veri seti içerisinde her kişinin seslerini içeren klasörler (ör: Kisi1, Kisi2, Kisi3)
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
print("Sınıf isimleri:", class_names)
label_map = {name: idx for idx, name in enumerate(class_names)}

# Her kişinin klasöründeki ses dosyalarını işleyip özellik ve etiketleri topluyoruz.
for person in class_names:
    person_dir = os.path.join(dataset_dir, person)
    for file_name in os.listdir(person_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(person_dir, file_name)
            feat = extract_features(file_path, n_mfcc=40)
            features.append(feat)
            labels.append(label_map[person])

features = np.array(features)
labels = np.array(labels)
print("Extracted features shape:", features.shape)
print("Labels shape:", labels.shape)

# 3. Verilerin Normalizasyonu
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Sınıflandırma Modelinin Tanımlanması
input_dim = features_scaled.shape[1]  # Örneğin 40
num_classes = len(class_names)

model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 5. Model Eğitimi
history = model.fit(features_scaled, labels,
                    epochs=100,
                    batch_size=8,
                    validation_split=0.2)

# 6. Eğitim Grafikleri: Loss ve Accuracy'nin PNG olarak kaydedilmesi
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig('loss_graph.png')
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('accuracy_graph.png')
plt.close()

print("Loss ve Accuracy grafikleri kaydedildi.")

# 7. Model, Scaler ve Label Haritasının Kaydedilmesi
model.save("voice_classification_model.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
print("Model, scaler ve label_map kaydedildi.")

# 8. MFCC Grafiklerinin Kaydedilmesi: Her kişinin MFCC görselleri, tek bir klasörde,
# dosya isimlerinde kişinin adını içerecek şekilde kaydedilsin.
mfcc_dir = "mfcc"
if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

for person in class_names:
    person_dir = os.path.join(dataset_dir, person)
    file_index = 1
    for file_name in os.listdir(person_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(person_dir, file_name)
            y, sr = librosa.load(file_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, sr=sr, x_axis='time')
            plt.colorbar()
            plt.title("MFCC - {} - {}".format(person, file_name))
            plt.tight_layout()
            # Dosya adını "person_index.png" şeklinde ayarlıyoruz.
            output_path = os.path.join(mfcc_dir, f"{person}_{file_index}.png")
            plt.savefig(output_path)
            plt.close()
            file_index += 1

print("MFCC görselleri, '{}' klasörüne kişi bazında ayrı dosyalar olarak kaydedildi.".format(mfcc_dir))
