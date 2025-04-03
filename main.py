import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
audio_file_path = '0 тревога.mp3'
audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
print("данные аудио:", audio_data)
print("частота дискретизации:", sample_rate)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio_data, sr=sample_rate)
plt.title('аудиосигнал')
plt.xlabel('время(с)')
plt.ylabel('амплитуда:')
plt.show()




