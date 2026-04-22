import numpy as np 
import matplotlib.pyplot as plt 

k = 2;  N = 1000; m = 2; M = 10; fb = 5; fa = 2; n = 3; omega = np.sqrt(k/m);
w0 = np.linspace(0, 4, 10000)

def get_amplitudes(k_idx):
    ksi = np.pi * k_idx / N
    jamma = 0.01;
    phi = ((w0/omega)**2 - 1 + np.exp(1j * ksi)) / ((w0/omega)**2 - 1 + np.exp(-1j * ksi))

    eq1 = (2*k - M*w0**2 + 2j*jamma*w0)*(np.exp(1j*ksi*n)+phi*np.exp(-1j*ksi*n))
    eq2 = k*(np.exp(1j*ksi*(n+1)) + phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*(n+1)) - np.exp(1j*ksi*n) + phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))
    eq3 = (2*k - m*w0**2 + 2j*jamma*w0)*(np.exp(1j*ksi*n)+phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))
    eq4 = k*(np.exp(1j*ksi*n) + phi*np.exp(-1j*ksi*n) + np.exp(1j*ksi*(n-1)) + phi*np.exp(-1j*ksi*(n-1)))

    B_val = (fa/2j + (eq3/eq2)*fb/2j)*(1/((eq1/eq2)*eq3 - eq4))
    A_val = -fb/(2j) * (1/eq2) + B_val * (eq1/eq2)
    
    return B_val, A_val

# Суммируем все моды для B и A
sum_B = np.zeros(len(w0), dtype=complex)
sum_A = np.zeros(len(w0), dtype=complex)

for ind in range(1, N):
    B_val, A_val = get_amplitudes(ind)
    sum_B += B_val
    sum_A += A_val

# Берем модуль и нормируем
amplitude_B = np.abs(sum_B)
amplitude_A = np.abs(sum_A)

amplitude_B_norm = amplitude_B / np.max(amplitude_B)
amplitude_A_norm = amplitude_A / np.max(amplitude_A)

# Строим оба графика вместе
plt.figure(figsize=(12, 6))
plt.plot(w0, amplitude_B_norm, 'b-', linewidth=2, label='Суммарная B (нормирована)')
plt.plot(w0, amplitude_A_norm, 'r-', linewidth=2, label='Суммарная A (нормирована)')
plt.xlabel(r'$\omega_0$', fontsize=12)
plt.ylabel(r'Нормированная амплитуда', fontsize=12)
plt.title(f'Суммарные амплитуды B и A (N={N} мод)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()