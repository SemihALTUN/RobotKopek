import os
import librosa
import soundfile as sf
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift

# Giriş ve çıkış dizinlerini belirleyin
input_dir = "C:\\Users\\semii\\OneDrive\\Masaüstü\\RobotKöpek\\Data"  # Orijinal ses dosyalarınızın bulunduğu dizin
output_dir = "C:\\Users\\semii\\OneDrive\\Masaüstü\\RobotKöpek\\VeriArttırma"  # Artırılmış ses dosyalarının kaydedileceği dizin
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Her bir ses dosyası için ayrı ayrı augmentation işlemlerini uygulayalım.
for file_name in os.listdir(input_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_dir, file_name)
        samples, sample_rate = librosa.load(file_path, sr=None)

        # Gürültü ekleme: p=1.0 ile işlemi zorunlu kılıyoruz.
        noise_augmenter = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        augmented_noise = noise_augmenter(samples=samples, sample_rate=sample_rate)
        noise_file_name = file_name.replace(".wav", "_noise.wav")
        noise_output_path = os.path.join(output_dir, noise_file_name)
        sf.write(noise_output_path, augmented_noise, sample_rate)
        print(f"{noise_output_path} kaydedildi.")

        # Time stretching: p=1.0 ile işlemi zorunlu kılıyoruz.
        stretch_augmenter = TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)
        augmented_stretch = stretch_augmenter(samples=samples, sample_rate=sample_rate)
        stretch_file_name = file_name.replace(".wav", "_stretch.wav")
        stretch_output_path = os.path.join(output_dir, stretch_file_name)
        sf.write(stretch_output_path, augmented_stretch, sample_rate)
        print(f"{stretch_output_path} kaydedildi.")

        # Pitch shifting: p=1.0 ile işlemi zorunlu kılıyoruz.
        pitch_augmenter = PitchShift(min_semitones=-2, max_semitones=2, p=1.0)
        augmented_pitch = pitch_augmenter(samples=samples, sample_rate=sample_rate)
        pitch_file_name = file_name.replace(".wav", "_pitch.wav")
        pitch_output_path = os.path.join(output_dir, pitch_file_name)
        sf.write(pitch_output_path, augmented_pitch, sample_rate)
        print(f"{pitch_output_path} kaydedildi.")
