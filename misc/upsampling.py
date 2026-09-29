import numpy as np
import matplotlib.pyplot as plt

N=128
F0=4

x_signal = [ np.sin(2*np.pi*F0 * n/N) for n in range(N)]
x_freq = np.fft.fft(x_signal)
x_ifft = np.fft.ifft(x_freq)
print(x_freq)

fig, axs = plt.subplots(3, 1)

ax = axs[0]
ax.plot(x_signal)
ax.set_xlim([0, N])
ax.grid(True)

ax = axs[1]
ax.plot(x_freq.real, label="real")
ax.plot(x_freq.imag, label="imag")
ax.plot(np.abs(x_freq), label="abs")
ax.set_xlim([0, N])
ax.grid(True)

ax = axs[2]
ax.plot(x_ifft.real)
ax.set_xlim([0, N])
ax.grid(True)

fig.savefig('sig.png')