import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
from numpy.fft import fft, fftshift

# ПАРАМЕТРЫ
a = 1
t1, t2 = 1, 3 
b = 0.4
T = 0.5

# ВРЕМЕННАЯ СЕТКА (t >= 0 !!!)
dt = 0.001
t = np.arange(0, 6, dt)

# СИГНАЛ g(t)
g = np.zeros_like(t)
g[(t >= t1) & (t <= t2)] = a

# ЗАШУМЛЁННЫЙ СИГНАЛ
xi = 2 * np.random.rand(len(t)) - 1
u = g + b * xi

# ФИЛЬТР ПЕРВОГО ПОРЯДКА
system = lti([1], [T, 1])

_, u_filt, _ = lsim(system, U=u, T=t)

# ФУРЬЕ
U = fftshift(fft(u))
UF = fftshift(fft(u_filt))
freq = fftshift(np.fft.fftfreq(len(t), dt))

# АЧХ
omega = 2 * np.pi * freq
H = 1 / np.sqrt((T * omega)**2 + 1)

# ГРАФИКИ
plt.figure()
plt.plot(t, g, label='g(t)')
plt.plot(t, u, label='u(t)', alpha=0.6)
plt.plot(t, u_filt, label='u_filt(t)')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(freq, np.abs(U), label='|U(ω)|')
plt.plot(freq, np.abs(UF), label='|U_filt(ω)|')
plt.xlim(-20, 20)
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(freq, H)
plt.xlim(-20, 20)
plt.title('АЧХ фильтра первого порядка')
plt.grid()
plt.show()

# ФУРЬЕ-ОБРАЗЫ
G = fftshift(fft(g))
U = fftshift(fft(u))
UF = fftshift(fft(u_filt))
freq = fftshift(np.fft.fftfreq(len(t), dt))
omega = 2 * np.pi * freq

# ЧАСТОТНАЯ ХАРАКТЕРИСТИКА ФИЛЬТРА
# W1(iω) = 1 / (1 + iωT)
W1 = 1 / np.sqrt((T * omega)**2 + 1)

# ГРАФИК 1: ВСЕ СПЕКТРЫ
plt.figure()
plt.plot(freq, np.abs(G), label='|G(ω)|')
plt.plot(freq, np.abs(U), label='|U(ω)|')
plt.plot(freq, np.abs(UF), label='|U_filt(ω)|')
plt.xlim(-20, 20)
plt.legend()
plt.title('Модули Фурье-образов сигналов')
plt.grid()
plt.show()

# ГРАФИК 2: ПРОВЕРКА W(iω)·U(ω)
UF_theory = W1 * np.abs(U)

plt.figure()
plt.plot(freq, np.abs(UF), label='|U_filt(ω)|')
plt.plot(freq, UF_theory, '--', label='|W1(iω)·U(ω)|')
plt.xlim(-20, 20)
plt.legend()
plt.title('Сравнение спектров фильтрованного сигнала')
plt.grid()
plt.show()


