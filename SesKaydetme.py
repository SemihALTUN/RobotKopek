import sounddevice as sd
import soundfile as sf

def record_and_save(duration=5, fs=16000, filename="AnnemDeneme.wav"):
    print("Kayıt başlıyor. Lütfen konuşmaya başlayın...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Kayıt bitene kadar bekler
    print("Kayıt tamamlandı, dosya kaydediliyor...")
    sf.write(filename, recording, fs)
    print(f"Dosya '{filename}' olarak kaydedildi.")

record_and_save()
