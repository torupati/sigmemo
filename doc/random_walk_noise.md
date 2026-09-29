# Random Walk

Random walk is usually introduced as a discrete random process, and that is a correct way. We can also define it as a continuous-time process whose time derivative is white noise.

$$
\frac{d}{dt}y(t) = x(t)
$$

Let $x(t)$ be white noise with noise density $\sigma$ (two-sided PSD $S_x(f) = \sigma^2$), as in the [previous section](./white_noise.md). In discrete time with sampling interval $T_s = 1/F_s$, the same process is the running sum

$$
y_n = y_{n-1} + T_s\,x_n
$$

## Variance

The steps are independent, so their variances add. With $\mathrm{Var}(x_n) = \sigma^2 F_s$,

$$
\mathrm{Var}(y_n) = n\,T_s^2\,\sigma^2 F_s = \sigma^2\,t, \qquad t = nT_s
$$

The standard deviation grows as $\sigma\sqrt{t}$, independent of the sampling rate. Because the variance changes with time, a random walk is non-stationary.

## Power Spectrum

Since the time derivative $\frac{d}{dt}$ becomes multiplication by $j2\pi f$ in the Fourier domain,

$$
j2\pi f\,Y(f) = X(f)
$$

Multiplying each side by its complex conjugate, the power is

$$
|Y(f)|^2 = \frac{|X(f)|^2}{4\pi^2f^2}
\qquad\Rightarrow\qquad
S_y(f) = \frac{\sigma^2}{(2\pi f)^2}
$$

For a sampled random walk, the running sum $1/(1-e^{-j2\pi f T_s})$ replaces the ideal integrator, and the exact PSD is

$$
S_y(f) = \frac{\sigma^2 T_s^2}{4\sin^2(\pi f T_s)}
$$

This agrees with $\sigma^2/(2\pi f)^2$ for $f \ll F_s$, and is about $\pi^2/4 \approx 2.5$ times larger at the Nyquist frequency.

As with white noise, the spectral density does not depend on the sampling rate; the sampling rate only limits the frequency range.

![PSD of white noise and random walk](./pictures/nb_psd_compare.png)

Note that a random walk is non-stationary, so the PSD above is what an averaged spectral estimate of finite records converges to. Use a window (e.g. Hann) when estimating it: without a window, the jump between the end and the start of the record leaks into all bins, and the estimate is about 2 times too high at every frequency while keeping the same $1/f^2$ slope (right panel).

See [White Noise and Random Walk — A Beginner's Guide](./noise_basics.md) for more figures and explanation.
