import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
a = '0 тревога.mp3'
b = '1 тревога.mp3'
c = '2 тревога.mp3'
d = '3 тревога.mp3'
e = '4 тревога.mp3'
f = '1 работы.mp3'
g = '2 работы.mp3'
h = '3 работы.mp3'
a, aa = librosa.load(a, sr=None)
b, bb = librosa.load(b, sr=None)
c, cc = librosa.load(c, sr=None)
d, ee = librosa.load(d, sr=None)
e, dd = librosa.load(e, sr=None)
f, ff = librosa.load(f, sr=None)
g, gg = librosa.load(g, sr=None)
h, hh = librosa.load(h, sr=None)
print("данные аудио:", a,b,c,d,e,f,g,h )
print("частота дискретизации:", aa,bb,cc,dd,ee,ff,gg,hh)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(a, sr=aa)
librosa.display.waveshow(b, sr=bb)
librosa.display.waveshow(c, sr=cc)
librosa.display.waveshow(d, sr=dd)
librosa.display.waveshow(e, sr=ee)
librosa.display.waveshow(f, sr=ff)
librosa.display.waveshow(g, sr=gg)
librosa.display.waveshow(h, sr=hh)
plt.title('аудиосигнал')
plt.xlabel('время(с)')
plt.ylabel('амплитуда:')
plt.show()




