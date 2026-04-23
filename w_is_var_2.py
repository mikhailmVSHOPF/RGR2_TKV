import numpy as np 
import matplotlib.pyplot as plt 

k = 2;  N = 4; m = 1; M = 5; fb = 1; fa = 2; n = 4; omega = np.sqrt(k/m);
w0 = np.linspace(0.001, 3, 10000)

def get_amplitudes(k_idx, w0):
    ksi = np.pi * k_idx / N
    jamma = 0.01;

    eq1 = 2*k - m*w0**2 - 2j*jamma*w0
    eq2 = 2*k - M*w0**2 - 2j*jamma*w0

    fa1 = fa/(2j*eq1); fb1 = fb/(2j* eq2)

    eq3 = k*(np.sin(ksi*n) + np.sin(ksi*(n-1)))/eq1

    eq4 = k*(np.sin(ksi*(n+1)) + np.sin(ksi*(n)))/eq2

    A_val = (fa1 + fb1*eq3)/(1-eq3*eq4)
    B_val = fb1 + A_val*eq4
    
    return B_val, A_val

# Суммируем все моды для B и A
sum_B = np.zeros(len(w0), dtype=complex)
sum_A = np.zeros(len(w0), dtype=complex)

for ind in range(1, N):
    B_val, A_val = get_amplitudes(ind, w0)
    sum_B += B_val
    sum_A += A_val

# Берем модуль и нормируем на максимум
amplitude_B = np.abs(sum_B)
amplitude_A = np.abs(sum_A)

amplitude_B_norm = amplitude_B   # нормировка B
amplitude_A_norm = amplitude_A   # нормировка A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Левый график - обрезанный сверху
percentile_95 = np.percentile(amplitude_B_norm, 95)
ax1.plot(w0, amplitude_B_norm, 'b-', linewidth=1.5)
ax1.set_ylim([0, percentile_95])
ax1.set_xlabel(r'$\omega_0$')
ax1.set_ylabel(r'Амплитуда')
ax1.set_title(f'Обрезано на {percentile_95:.2f}')
ax1.grid(True, alpha=0.3)

# Правый график - логарифмический
ax2.plot(w0, amplitude_B_norm, 'r-', linewidth=1.5)
ax2.set_yscale('log')
ax2.set_ylim([1e-4, None])
ax2.set_xlabel(r'$\omega_0$')
ax2.set_ylabel(r'Амплитуда (лог)')
ax2.set_title('Логарифмическая шкала')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()