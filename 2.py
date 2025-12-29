import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim, TransferFunction

# Загрузка данных из TXT-файла
file_path = r"C:\Users\sasha\Downloads\SBER_251228_251229.txt"
data = pd.read_csv(file_path, delimiter=",")

# Используем цену закрытия
price = data["<CLOSE>"].values
t = np.arange(len(price))  # дискретное время

# Постоянные времени фильтра (в шагах данных)
T_values = {
    "1 день": 1,
    "1 неделя": 5,
    "1 месяц": 22,
    "3 месяца": 66,
    "1 год": 252
}

plt.figure(figsize=(14, 10))

plot_index = 1
for label, T in T_values.items():
    # Передаточная функция W(p) = 1 / (T p + 1)
    system = TransferFunction([1], [T, 1])

    # Начальное условие — первое значение сигнала
    _, y, _ = lsim(system, U=price, T=t, X0=[price[0]])

    plt.subplot(3, 2, plot_index)
    plt.plot(t, price, label="Исходный сигнал", linewidth=1)
    plt.plot(t, y, label=f"Сглаженный сигнал (T = {label})", linewidth=2)
    plt.title(f"Сглаживание при T = {label}")
    plt.xlabel("Время")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)

    plot_index += 1

plt.tight_layout()
plt.show()
