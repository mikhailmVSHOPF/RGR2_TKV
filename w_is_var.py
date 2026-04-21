import numpy as np 
import matplotlib.pyplot as plt 

k = 2; n = 5; m = 1; M = 2; fb = 1; fa = 2; N = 10; omega = np.sqrt(k/m); num = 10000;
result = np.zeros(num)
w0 = np.linspace(0,10,num)

ksi = np.pi / N
def Ampl(k):
    ksi = np.pi * k / N
    phi = ((w0/omega)**2 - 1 + np.exp(1j * ksi)) / ((w0/omega)**2 - 1 + np.exp(-1j * ksi))

    eq_1_part_1 = k*(np.exp(1j*ksi*(n+1)) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*(n+1)) + np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))
    eq_1_part_2 = (2*k - M*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n))

    eq_2_part_1 = k*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n) + np.exp(1j*ksi*(n-1)) - phi*np.exp(-1j*ksi*(n-1)))
    eq_2_part_2 = (2*k - m*w0**2)*(np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi)*np.exp(-1j*ksi*n))

    B = (fb/(2j) * (eq_2_part_2/eq_1_part_1) - fa/(2j)) * (1/(eq_2_part_1 - eq_2_part_2*(eq_1_part_2/eq_1_part_1)))
    A = eq_1_part_2/eq_1_part_1 * B + fb/(1j*eq_1_part_1)
    return B, A , (eq_2_part_1 - eq_2_part_2*(eq_1_part_2/eq_1_part_1))

B_1, A_1, C = Ampl(1)
B_2, A_2, _ = Ampl(2)
B_3, A_3, _ = Ampl(3)
B_4, A_4, _ = Ampl(4)
B_5, A_5, _ = Ampl(5)
B_6, A_6, _ = Ampl(6)
B_7, A_7, _ = Ampl(7)
B_8, A_8, _ = Ampl(8)
B_9, A_9, _ = Ampl(9)
B_10, A_10, _ = Ampl(10)


# Вычисляем данные
mod_B = np.abs(B_1)
mod_B_2 = np.abs(B_2)
mod_B_3 = np.abs(B_3)
mod_B_4 = np.abs(B_4)
mod_B_5 = np.abs(B_5)
mod_B_6 = np.abs(B_6)
mod_B_7 = np.abs(B_7)
mod_B_8 = np.abs(B_8)
mod_B_9 = np.abs(B_9)
mod_B_10 = np.abs(B_10)

mod_A = np.abs(A_1)
mod_A_2 = np.abs(A_2)
mod_A_3 = np.abs(A_3)
mod_A_4 = np.abs(A_4)


# Находим разумные пределы (отсекаем выбросы)
percentile = 99  # используем 99-й перцентиль
ylim1_max = np.percentile(mod_B[mod_B < np.inf], percentile)
#ylim2_max = np.percentile(mod_B_squared[mod_B_squared < np.inf], percentile)


# # Модуль |B|
# plt.plot(w0, (mod_B + mod_B_2 + mod_B_3 + mod_B_4 + mod_B_5 + mod_B_6 + mod_B_7 + mod_B_8 + mod_B_9)/100, linewidth=2)
# plt.plot(w0, mod_B_2, linewidth=2)

# plt.plot(w0, mod_A, linewidth=2)
plt.plot(w0, mod_B_5, linewidth=2)

plt.ylim(0, ylim1_max)  # ← ограничение по Y
plt.xlabel(r'w', fontsize=12)
plt.ylabel(r'$|B|$', fontsize=12)
plt.title('Модуль B', fontsize=14)
plt.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()
