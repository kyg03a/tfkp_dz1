import numpy as np
import matplotlib.pyplot as plt


# СИГНАЛЫ
T = 20
N = 4096
t = np.linspace(0, T, N)
dt = t[1] - t[0]

g = np.sin(2*np.pi*1*t)                     # исходный сигнал
noise = 0.6*np.sin(2*np.pi*8*t)             # шум
u = g + noise                               # зашумленный

# FFT
freq = np.fft.fftshift(np.fft.fftfreq(N, dt))
U = np.fft.fftshift(np.fft.fft(u))

# ПАРАМЕТРЫ ФИЛЬТРА
c = 2.0      # параметр b1
d = 8.0      # частота подавления ω0

W2 = (-freq**2 + d**2) / (-freq**2 + 1j*c*freq + d**2)

# ФИЛЬТРАЦИЯ
UF = W2 * U
u_filt = np.real(np.fft.ifft(np.fft.ifftshift(UF)))

# 1. ВРЕМЕННЫЕ СИГНАЛЫ
plt.figure()
plt.plot(t, g, label='g(t)')
plt.plot(t, u, label='u(t)', alpha=0.6)
plt.plot(t, u_filt, label='u_filt(t)')
plt.xlim(0, 5)
plt.legend()
plt.grid()
plt.title('Сравнение сигналов во времени')
plt.show()

# 2. АЧХ ФИЛЬТРА
plt.figure()
plt.plot(freq, np.abs(W2))
plt.xlim(0, 20)
plt.grid()
plt.title('АЧХ фильтра W2(iω)')
plt.show()

# 3. МОДУЛИ СПЕКТРОВ
plt.figure()
plt.plot(freq, np.abs(np.fft.fftshift(np.fft.fft(g))), label='|ĝ(ω)|')
plt.plot(freq, np.abs(U), label='|û(ω)|')
plt.plot(freq, np.abs(UF), label='|û_filt(ω)|')
plt.xlim(0, 20)
plt.legend()
plt.grid()
plt.title('Модули Фурье-образов')
plt.show()

# 4. ПРОВЕРКА ЧЕРЕЗ ЧАСТОТУ
u_ifft = np.real(np.fft.ifft(np.fft.ifftshift(W2 * U)))

plt.figure()
plt.plot(t, u_filt, label='Фильтрация во времени')
plt.plot(t, u_ifft, '--', label='IFFT(W2·Û)')
plt.xlim(0, 5)
plt.legend()
plt.grid()
plt.title('Сравнение способов фильтрации')
plt.show()
