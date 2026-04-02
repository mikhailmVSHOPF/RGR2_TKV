import numpy as np 
import matplotlib.pyplot as plt 
fb = 1; w0 = 1; m = 1; M = 1; ksi = 1; N = 10; n = 10; k = 1; fa = 1

def A(w0):
    return (1/(2j)) * (fb*k/(2*k - w0**2 * m) + fa) * (1 / (2 - (2*k * (1 - w0**2 * M / k))*np.sin(ksi*(n + 1)) - w0**2 * M - k**2 * (1+2*np.sin(ksi*(n-1)))/(2*k - w0**2 * m)))

def B(n, A_val):
    return k*A_val*(1+2*np.sin(ksi*(n-1)))/(2*k - w0**2 * m) + k * fb * (1/(2j*(2*k - w0**2 * m)))

w0 = np.linspace(0, 10, 1000)
A_values = A(w0)
A_real = A_values.real

plt.plot(w0, A_real); plt.show()