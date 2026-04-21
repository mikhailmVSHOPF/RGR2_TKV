import numpy as np 
import matplotlib.pyplot as plt 

k = 10; n = 4; m = 1; M = 2; fb = 1; fa = 2; N = 10; omega = np.sqrt(k/m); w0 = 1;

# Создаем ksi с нужным диапазоном
ksi_max = np.pi  # период cos²
q = np.linspace(0, 1000, 100000)
ksi_full = np.pi * np.array(q) / N

# Создаем маску для обрезки
mask = ksi_full <= ksi_max

# Применяем маску к ksi_full
ksi = ksi_full[mask]

# Теперь все остальные расчеты делаем с обрезанным ksi
phi = ((w0/omega)**2 - 1 + np.exp(1j * ksi)) / ((w0/omega)**2 - 1 + np.exp(-1j * ksi))

eq_1_part_1 = k*(np.exp(1j*ksi*(n+1)) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*(n+1)) + 
                 np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))
eq_1_part_2 = (2*k - M*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n))

eq_2_part_1 = k*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n) + 
                 np.exp(1j*ksi*(n-1)) - phi*np.exp(-1j*ksi*(n-1)))
eq_2_part_2 = (2*k - m*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))

B = (fb/(2j) * (eq_2_part_2/eq_1_part_1) - fa/(2j)) * (1/(eq_2_part_1 - eq_2_part_2*(eq_1_part_2/eq_1_part_1)))

mod_B = np.abs(B)
mod_B_squared = np.abs(B)**2

# Находим разумные пределы (отсекаем выбросы)
percentile = 99
ylim1_max = np.percentile(mod_B[mod_B < np.inf], percentile)

# Построение
plt.figure(figsize=(10, 6))
plt.plot(ksi, mod_B_squared, 'b-', linewidth=2)
plt.ylim(0, ylim1_max)
plt.xlabel(r'$\xi$', fontsize=12)
plt.ylabel(r'$|B|$', fontsize=12)
plt.title(f'Модуль B (ξ ∈ [0, {ksi_max:.2f}])', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()