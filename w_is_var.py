import numpy as np 
import matplotlib.pyplot as plt 

k = 1; n = 5; m = 1; M = 2; fb = 1; fa = 2; N = 1000; omega = np.sqrt(k/m);

w0 = np.linspace(0,1,10000)
ksi = np.pi / N

phi = ((w0/omega)**2 - 1 + np.exp(1j * ksi)) / ((w0/omega)**2 - 1 + np.exp(-1j * ksi))

eq_1_part_1 = k*(np.exp(1j*ksi*(n+1)) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*(n+1)) + np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))
eq_1_part_2 = (2*k - M*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n))

eq_2_part_1 = k*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n) + np.exp(1j*ksi*(n-1)) - phi*np.exp(-1j*ksi*(n-1)))
eq_2_part_2 = (2*k - m*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))

B = (fb/(2j) * (eq_2_part_2/eq_1_part_1) - fa/(2j)) * (1/(eq_2_part_1 - eq_2_part_2*(eq_1_part_2/eq_1_part_1)))

# Вычисляем данные
mod_B = np.abs(B)
mod_B_squared = np.abs(B)**2

# Находим разумные пределы (отсекаем выбросы)
percentile = 99  # используем 99-й перцентиль
ylim1_max = np.percentile(mod_B[mod_B < np.inf], percentile)
ylim2_max = np.percentile(mod_B_squared[mod_B_squared < np.inf], percentile)


# Модуль |B|
plt.plot(w0, mod_B, 'b-', linewidth=2)
plt.ylim(0, ylim1_max)  # ← ограничение по Y
plt.xlabel(r'w', fontsize=12)
plt.ylabel(r'$|B|$', fontsize=12)
plt.title('Модуль B', fontsize=14)
plt.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()
