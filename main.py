import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def load_audio_files(directory):
    """Рекурсивно загружает MP3-файлы и возвращает их характеристики."""
    file_names = []
    audios = []
    sampling_rates = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp3'):
                file_path = os.path.join(root, file)
                try:
                    data, sr = librosa.load(file_path, sr=None)
                    file_names.append(file_path)
                    audios.append(data)
                    sampling_rates.append(sr)
                except Exception as e:
                    print(f"Ошибка при загрузке {file_path}: {e}")
    return file_names, audios, sampling_rates

def plot_audio(data, sr, title="Аудиосигнал"):
    """Визуализация волновой формы сигнала."""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.show()

def extract_features_deep(audio, sr):
    """Извлекает расширенный набор аудио-признаков."""
    features = {}
    
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spec_centroid)
    features['spectral_centroid_std'] = np.std(spec_centroid)
    
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)

    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features['rms'] = np.mean(librosa.feature.rms(y=audio))

    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spec_contrast, axis=1)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    features['spectral_entropy_mean'] = np.mean(spectral_flatness)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
    
    return features

def predict_sound_deep(features):
    """Улучшенная эвристическая модель классификации."""
    score = 0

    if features['spectral_centroid_mean'] > 3500:
        score += 1
    if features['spectral_rolloff_mean'] > 5000:
        score += 1
    if features['rms'] > 0.03:
        score += 1
    if np.mean(features['spectral_contrast_mean']) > 15:
        score += 1

    if features['mfcc_mean'][0] > -300:
        score += 1
    if features['mfcc_mean'][1] > 100:
        score += 1
    
    print(f"  Композитный скор: {score}")
    return "Тревога" if score >= 3 else "Дорожные работы"

def print_deep_features(features):
    """Выводит все вычисленные признаки с новыми полями."""
    print("Вычисленные признаки:")
    print(f"  Спектральный центр (среднее): {features['spectral_centroid_mean']:.2f}")
    print(f"  Спектральный центр (STD): {features['spectral_centroid_std']:.2f}")
    print(f"  Спектральный rolloff (среднее): {features['spectral_rolloff_mean']:.2f}")
    print(f"  Спектральная ширина: {features['spectral_bandwidth_mean']:.2f}")
    print(f"  Zero Crossing Rate: {features['zero_crossing_rate']:.4f}")
    print(f"  RMS энергия: {features['rms']:.4f}")
    print(f"  Спектральная энтропия: {features['spectral_entropy_mean']:.2f}")
    print(f"  Хрома-признаки: {features['chroma_mean']:.2f}")
    
    print("  Спектральный контраст (средние):")
    for i, contrast in enumerate(features['spectral_contrast_mean']):
        print(f"    Коэффициент {i+1}: {contrast:.2f}")
        
    print("  MFCC коэффициенты (средние):")
    for i, coef in enumerate(features['mfcc_mean']):
        print(f"    MFCC {i+1}: {coef:.2f}")
    
    print("  MFCC (стандартные отклонения):")
    for i, std in enumerate(features['mfcc_std']):
        print(f"    MFCC {i+1} (STD): {std:.2f}")

def main():
    directory = input("Введите путь к директории с MP3 файлами: ").strip()
    if not os.path.isdir(directory):
        print("Директория не существует!")
        return

    file_names, audios, srs = load_audio_files(directory)
    if not file_names:
        print("MP3 файлы не найдены!")
        return

    dataset_features = []
    dataset_labels = []
    
    for file_path, audio, sr in zip(file_names, audios, srs):
        print(f"\nОбработка файла: {file_path}")
        
        features = extract_features_deep(audio, sr)
        print_deep_features(features)

        dataset_features.append(list(features.values()))
        dataset_labels.append("Тревога" if "тревога" in file_path.lower() else "Дорожные работы")
        
        prediction = predict_sound_deep(features)
        print(f"Результат анализа: {prediction}")
        print("Сигнал тревоги!" if prediction == "Тревога" else "Всё в порядке.")

if __name__ == "__main__":
    main()