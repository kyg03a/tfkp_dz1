import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift

# ПАРАМЕТРЫ СИГНАЛА
a = 1
t1, t2 = -1, 1
b = 0.4
c = 0

# ВРЕМЕННАЯ СЕТКА
dt = 0.001
Tmax = 5
t = np.arange(-Tmax, Tmax, dt)

# СИГНАЛ g(t)
g = np.zeros_like(t)
g[(t >= t1) & (t <= t2)] = a

# ЗАШУМЛЁННЫЙ СИГНАЛ
xi = 2 * np.random.rand(len(t)) - 1
u = g + b * xi

# FFT
U = fftshift(fft(u))
freq = fftshift(np.fft.fftfreq(len(t), dt))
omega = 2 * np.pi * freq

# ЧАСТОТНЫЙ ФИЛЬТР (НИЗКИХ ЧАСТОТ)
omega_c = 5
W = 1 / (1 + (omega / omega_c)**2)

# ФИЛЬТРАЦИЯ В ЧАСТОТНОЙ ОБЛАСТИ
U_filt = U * W
u_filt = np.real(ifft(ifftshift(U_filt)))

plt.figure(figsize=(10, 5))
plt.plot(t, u, label='u(t)')
plt.plot(t, u_filt, label='u_filt(t)')
plt.legend()
plt.title('Задание 1: фильтрация в частотной области')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(freq, np.abs(U), label='|U(ω)|')
plt.plot(freq, np.abs(U_filt), label='|U_filt(ω)|')
plt.xlim(-20, 20)
plt.legend()
plt.title('Амплитудные спектры')
plt.grid()
plt.show()
