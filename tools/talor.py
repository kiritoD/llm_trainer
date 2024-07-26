import numpy as np


def f(x):
    return np.sin(x)


a = np.pi / 4
f_prime = f(a + np.diff(np.linspace(a - 1e-5, a + 1e-5, 100))) / (
    np.diff(np.linspace(a - 1e-5, a + 1e-5, 100))[0]
)

f_double_prime = f_prime(a + np.diff(np.linspace(a - 1e-5, a + 1e-5, 100))) / (
    np.diff(np.linspace(a - 1e-5, a + 1e-5, 100))[0]
)

import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 1000)
plt.plot(x, f(x), label="f(x)")
plt.plot(x, f_prime(x), label="f(x)")
plt.plot(x, f_double_prime(x), label="f(x)")
plt.legend()
plt.show()
