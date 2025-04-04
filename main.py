import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

def load_audio_files(directory):
    file_names = []
    audios = []
    sampling_rates = []
    
    for file in os.listdir(directory):
        if file.lower().endswith('.mp3'):
            file_path = os.path.join(directory, file)
            try:
                data, sr = librosa.load(file_path, sr=None)
                file_names.append(file)
                audios.append(data)
                sampling_rates.append(sr)
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}")

    return file_names, audios, sampling_rates

def plot_audio(data, sr, title="Аудиосигнал"):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.show()

def main():

    directory = input("Введите путь к директории с .mp3 файлами: ").strip()
    if not os.path.isdir(directory):
        print("Указанная директория не существует.")
        return
    
    file_names, audios, srs = load_audio_files(directory)
    if not file_names:
        print("В директории не найдено .mp3 файлов.")
        return
    
    for name, audio, sr in zip(file_names, audios, srs):
        print(f"\nОбработка файла: {name}")
        plot_audio(audio, sr, title=f"Вейвформа: {name}")

if __name__ == "__main__":
    main()