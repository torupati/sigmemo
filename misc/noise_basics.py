"""Figures for doc/noise_basics.md (white noise and random walk for beginners).

Convention: sigma is the noise density [V/sqrt(Hz)] of a TWO-SIDED PSD,
    S_x(f) = sigma^2  for -fs/2 < f < fs/2.
Discrete white noise:  x[n] = sigma * sqrt(fs) * w[n],  w[n] ~ N(0, 1)
Random walk:           y[n] = Ts * sum_{k<=n} x[k]  (so that dy/dt = x)

Run:  uv run python misc/noise_basics.py
Output: doc/pictures/nb_*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "doc" / "pictures"

# Colors: white noise = blue, random walk = orange (same entity -> same color in every figure).
C_WN = "#2a78d6"
C_RW = "#eb6834"
C_ALT = "#1baf7a"
C_THEORY = "#0b0b0b"
C_TEXT2 = "#52514e"
C_PATH = "#b8b7b0"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#0b0b0b",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3de",
        "grid.linewidth": 0.8,
        "xtick.color": C_TEXT2,
        "ytick.color": C_TEXT2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)

SIGMA = 0.1  # noise density [V/sqrt(Hz)]
rng = np.random.default_rng(20260929)


def white_noise(n: int, fs: float, sigma: float = SIGMA) -> np.ndarray:
    return sigma * np.sqrt(fs) * rng.standard_normal(n)


def random_walk(x: np.ndarray, fs: float) -> np.ndarray:
    return np.cumsum(x, axis=-1) / fs


def averaged_psd(x: np.ndarray, fs: float, nseg: int, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Two-sided PSD [V^2/Hz] at f >= 0, averaged over segments (Welch without overlap).

    x may be 1-D (split into segments) or 2-D (one segment per row).
    """
    segs = x[: x.size // nseg * nseg].reshape(-1, nseg) if x.ndim == 1 else x
    w = np.hanning(nseg) if window == "hann" else np.ones(nseg)
    segs = segs - segs.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(segs * w, axis=1)) ** 2 / (fs * np.sum(w**2))
    # Bins 0 and 1 are biased by the mean removal (bin 1 overlaps DC under the Hann window): drop them.
    return np.fft.rfftfreq(nseg, 1.0 / fs)[2:], spec.mean(axis=0)[2:]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"saved {OUT_DIR / name}")


# ---------------------------------------------------------------------------
# Figure 1: a random walk is the running sum of white noise
# ---------------------------------------------------------------------------
def fig_construction() -> None:
    fs, n = 10.0, 60
    t = np.arange(n) / fs
    x = white_noise(n, fs)
    y = random_walk(x, fs)

    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    ax = axs[0]
    ax.vlines(t, 0, x, color=C_WN, linewidth=2)
    ax.plot(t, x, "o", color=C_WN, markersize=4)
    ax.axhline(0, color=C_TEXT2, linewidth=0.8)
    ax.set_title("White noise x[n]: every sample is a new, independent random number")
    ax.set_ylabel("x [V]")

    ax = axs[1]
    ax.step(t, y, where="post", color=C_RW)
    ax.plot(t, y, "o", color=C_RW, markersize=4)
    ax.axhline(0, color=C_TEXT2, linewidth=0.8)
    ax.set_title("Random walk y[n] = y[n-1] + x[n]·Ts: add up the white noise steps")
    ax.set_ylabel("y [V·s]")
    ax.set_xlabel("time [s]")
    fig.tight_layout()
    save(fig, "nb_construction.png")


# ---------------------------------------------------------------------------
# Figure 2: four views of white noise
# ---------------------------------------------------------------------------
def fig_white_noise_views() -> None:
    fs, n = 1000.0, 2**16
    x = white_noise(n, fs)
    t = np.arange(n) / fs
    std_theory = SIGMA * np.sqrt(fs)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axs[0, 0]
    m = int(0.2 * fs)
    ax.plot(t[:m], x[:m], color=C_WN, linewidth=1.0)
    ax.axhline(std_theory, color=C_THEORY, linestyle="--", linewidth=1.2, label=r"$\pm\sigma\sqrt{f_s}$ (1 std)")
    ax.axhline(-std_theory, color=C_THEORY, linestyle="--", linewidth=1.2)
    ax.set_ylim(-5 * std_theory, 5 * std_theory)
    ax.legend(loc="upper right")
    ax.set_title("(a) Time domain: no pattern, no memory")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x [V]")

    ax = axs[0, 1]
    ax.hist(x, bins=80, density=True, color=C_WN, alpha=0.55, edgecolor="white", linewidth=0.5)
    v = np.linspace(-4.5 * std_theory, 4.5 * std_theory, 400)
    pdf = np.exp(-0.5 * (v / std_theory) ** 2) / (np.sqrt(2 * np.pi) * std_theory)
    ax.plot(v, pdf, color=C_THEORY, linewidth=1.5, label=r"Gaussian, std $=\sigma\sqrt{f_s}$")
    ax.set_title("(b) Histogram: amplitude distribution")
    ax.set_xlabel("x [V]")
    ax.set_ylabel("probability density [1/V]")
    ax.legend(loc="upper right")

    ax = axs[1, 0]
    lags = np.arange(-20, 21)
    xc = x - x.mean()
    r = np.array([np.dot(xc[: n - abs(k)], xc[abs(k) :]) / n for k in lags]) / np.var(x)
    ax.vlines(lags, 0, r, color=C_WN, linewidth=2)
    ax.plot(lags, r, "o", color=C_WN, markersize=4)
    ax.axhline(0, color=C_TEXT2, linewidth=0.8)
    ax.set_title("(c) Autocorrelation: correlated only with itself (lag 0)")
    ax.set_xlabel("lag k [samples]")
    ax.set_ylabel(r"$\rho[k] = R[k]/R[0]$")
    ax.set_ylim(-0.15, 1.1)

    ax = axs[1, 1]
    f1, p1 = averaged_psd(x[:4096], fs, 4096, window="rect")
    f2, p2 = averaged_psd(x, fs, 1024)
    ax.plot(f1, p1, color=C_WN, alpha=0.3, linewidth=0.7, label="one periodogram (4096 pts)")
    ax.plot(f2, p2, color=C_WN, linewidth=2, label="average of 64 segments")
    ax.axhline(SIGMA**2, color=C_THEORY, linestyle="--", linewidth=1.5, label=r"theory $\sigma^2$")
    ax.set_yscale("log")
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(SIGMA**2 / 300, SIGMA**2 * 30)
    ax.set_title("(d) Power spectral density: flat")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [V²/Hz]")
    ax.legend(loc="upper right")

    fig.suptitle(
        rf"Four views of the same white noise   ($\sigma$ = {SIGMA} V/$\sqrt{{Hz}}$, $f_s$ = {fs:.0f} Hz)",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "nb_white_noise_views.png")


# ---------------------------------------------------------------------------
# Figure 3: same noise density, different sampling rate
# ---------------------------------------------------------------------------
def fig_sampling_rate() -> None:
    cases = ((100.0, C_WN), (1000.0, C_ALT))
    duration = 655.36

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for fs, color in cases:
        n = int(duration * fs)
        x = white_noise(n, fs)
        m = int(0.5 * fs)
        axs[0].plot(np.arange(m) / fs, x[:m], color=color, linewidth=1.0, zorder=3 if fs < 500 else 2,
                    label=rf"$f_s$={fs:.0f} Hz, std = {np.std(x):.2f} V")
        f, p = averaged_psd(x, fs, 256 if fs < 500 else 2048)
        axs[1].plot(f, p, color=color, label=rf"$f_s$={fs:.0f} Hz (up to {fs / 2:.0f} Hz)")

    axs[0].set_title("Time domain: faster sampling → larger amplitude")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("x [V]")
    axs[0].legend(loc="upper right")

    axs[1].axhline(SIGMA**2, color=C_THEORY, linestyle="--", linewidth=1.5, label=r"$\sigma^2$")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_ylim(SIGMA**2 / 10, SIGMA**2 * 10)
    axs[1].set_title("PSD: same level, wider band")
    axs[1].set_xlabel("frequency [Hz]")
    axs[1].set_ylabel("PSD [V²/Hz]")
    axs[1].legend(loc="upper left")

    fig.suptitle(rf"Same noise density $\sigma$ = {SIGMA} V/$\sqrt{{Hz}}$, two sampling rates", fontsize=12)
    fig.tight_layout()
    save(fig, "nb_white_noise_sampling_rate.png")


# ---------------------------------------------------------------------------
# Figure 4: random walk spreads like sqrt(t)
# ---------------------------------------------------------------------------
def fig_random_walk_spread() -> None:
    fs, duration, n_paths = 100.0, 100.0, 1000
    n = int(fs * duration)
    t = np.arange(1, n + 1) / fs
    y = random_walk(white_noise(n_paths * n, fs).reshape(n_paths, n), fs)
    env = SIGMA * np.sqrt(t)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axs[0]
    for k in range(40):
        ax.plot(t, y[k], color=C_PATH, linewidth=0.6)
    ax.plot(t, y[0], color=C_RW, linewidth=1.5, label="one realization")
    ax.plot(t, env, color=C_THEORY, linestyle="--", linewidth=1.5, label=r"$\pm\sigma\sqrt{t}$ (68 %)")
    ax.plot(t, -env, color=C_THEORY, linestyle="--", linewidth=1.5)
    ax.plot(t, 2 * env, color=C_THEORY, linestyle=":", linewidth=1.5, label=r"$\pm 2\sigma\sqrt{t}$ (95 %)")
    ax.plot(t, -2 * env, color=C_THEORY, linestyle=":", linewidth=1.5)
    ax.set_title("40 random walks from the same noise source")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("y [V·s]")
    ax.legend(loc="upper left")

    ax = axs[1]
    idx = np.unique(np.logspace(0, np.log10(n), 40).astype(int)) - 1
    ax.plot(t[idx], y[:, idx].std(axis=0), "o", color=C_RW, markersize=5,
            label=f"measured std over {n_paths} walks")
    ax.plot(t, env, color=C_THEORY, linestyle="--", linewidth=1.5, label=r"theory $\sigma\sqrt{t}$ (slope 1/2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Spread grows as √t (log-log)")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("standard deviation of y [V·s]")
    ax.legend(loc="upper left")

    fig.suptitle(rf"Random walk from white noise of $\sigma$ = {SIGMA} V/$\sqrt{{Hz}}$, $f_s$ = {fs:.0f} Hz",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "nb_random_walk_spread.png")


# ---------------------------------------------------------------------------
# Figure 5: PSD of white noise vs random walk
# ---------------------------------------------------------------------------
def fig_psd_compare() -> None:
    fs, nseg, n_avg = 100.0, 4096, 200
    ts = 1.0 / fs
    x = white_noise(n_avg * nseg, fs).reshape(n_avg, nseg)
    y = random_walk(x, fs)
    f, px = averaged_psd(x, fs, nseg)
    _, py = averaged_psd(y, fs, nseg)
    _, py_rect = averaged_psd(y, fs, nseg, window="rect")
    rw_cont = SIGMA**2 / (2 * np.pi * f) ** 2
    rw_disc = SIGMA**2 * ts**2 / (4 * np.sin(np.pi * f * ts) ** 2)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ax.plot(f, px, color=C_WN, label="white noise x (simulated)")
    ax.plot(f, py, color=C_RW, label="random walk y (simulated)")
    ax.plot(f, np.full_like(f, SIGMA**2), color=C_THEORY, linestyle="--", linewidth=1.5,
            label=r"$\sigma^2$")
    ax.plot(f, rw_cont, color=C_THEORY, linestyle=":", linewidth=1.8, label=r"$\sigma^2/(2\pi f)^2$ (slope −2)")
    ax.plot(f, rw_disc, color=C_THEORY, linestyle="-.", linewidth=1.0,
            label=r"$\sigma^2 T_s^2 / (4\sin^2(\pi f T_s))$ (sampled)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(f[0], fs / 2)
    ax.set_title("PSD: flat vs. falling as 1/f²")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [V²/Hz] (x) and [V²·s²/Hz] (y)")
    ax.legend(loc="lower left")

    ax = axs[1]
    ax.plot(f, py_rect / rw_disc, color=C_TEXT2, linewidth=1.2, label="no window (rectangular)")
    ax.plot(f, py / rw_disc, color=C_RW, linewidth=1.2, label="Hann window")
    ax.plot(f, rw_cont / rw_disc, color=C_THEORY, linestyle=":", linewidth=1.8,
            label=r"continuous formula $\sigma^2/(2\pi f)^2$")
    ax.axhline(1.0, color=C_THEORY, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(f[0], fs / 2)
    ax.set_ylim(0, 3)
    ax.set_title("Pitfall: estimate ÷ exact discrete theory")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper left")

    fig.suptitle(rf"$\sigma$ = {SIGMA} V/$\sqrt{{Hz}}$, $f_s$ = {fs:.0f} Hz, average of {n_avg} records of {nseg} samples",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "nb_psd_compare.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_construction()
    fig_white_noise_views()
    fig_sampling_rate()
    fig_random_walk_spread()
    fig_psd_compare()
