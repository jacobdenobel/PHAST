import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
from matplotlib.colors import LogNorm, Normalize

from .defaults import DEFAULT_BIN_TO_LOC_MAP


def virtual_channel_frequencies(n_channels: int, max_freq: int = None):
    fft_bin = np.fft.fftfreq(256, 1 / 17400)[:128].clip(0, max_freq)
    return np.interp(np.linspace(0, 15, n_channels), DEFAULT_BIN_TO_LOC_MAP, fft_bin)


def frequency_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_yscale("symlog", linthresh=1000.0, base=2)
    ax.yaxis.set_major_formatter(mplticker.ScalarFormatter())
    ax.yaxis.set_major_locator(
        mplticker.SymmetricalLogLocator(ax.yaxis.get_transform())
    )
    ax.yaxis.set_label_text("frequency [Hz]")


def time_vs_freq(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("time [s]")
    frequency_ax(ax)


def plot_heatmap(
    t, y, data, 
    ax=None, fig=None, 
    show_bands: bool = True, 
    pad_idx: bool = False,
    figsize=(9, 4),
    logcolors: bool = False
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    norm = LogNorm(vmin=data.min(), vmax=data.max()) if logcolors else Normalize(vmin=data.min(), vmax=data.max())

    if pad_idx:
        n_idx = np.nonzero(data.sum(axis=0))[0]
        n_idx = np.unique(np.c_[n_idx - 1, n_idx, n_idx + 1].ravel())
        n_idx = n_idx[n_idx < t.size]
        img = ax.pcolormesh(
            t[n_idx], y, data[:, n_idx], cmap="inferno", norm=norm
            
        )
    else:
        img = ax.pcolormesh(
            t[:], y, data[:, :], cmap="inferno", norm=norm
        )
    time_vs_freq(ax)
    ax.set_xlabel("time [s]")
    fig.colorbar(img, ax=ax)
    
    if show_bands:
        for f in y:
            ax.plot([0, t[-1]], [f, f], color="white", alpha=0.3)
        ax.set_xlim(0, t[-1])


def min_max_scale(data, a=-80, b=0):
    data_min = np.min(data)
    data_max = np.max(data)
    return a + (data - data_min) * (b - a) / (data_max - data_min)


def make_bins(n, data):
    return data[:, : len(data[0]) // n * n].reshape(data.shape[0], -1, n).sum(axis=2)


def smooth(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None
) -> np.ndarray:
    
    hop_length = hop_length or window_size // 4
    window = scipy.signal.get_window(window_type, window_size)
    wsum = window.sum()
    data = np.vstack(
        [
            (np.convolve(data[i], window, mode="same") / wsum)[::hop_length]
            for i in range(data.shape[0])
        ]
    )
    return data

def apply_filter(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None,
    scale: bool = True,
    clip_outliers: float = 0.0,
    n_bins: int = 0,
    resample_to: int = None,
):
    data = data.copy()
    if clip_outliers != 0.0:
        q99 = np.quantile(data.ravel(), clip_outliers)
        data[data > q99] = q99

    if n_bins != 0:
        data = make_bins(n_bins, data)
        
    data = smooth(data, window_type, window_size, hop_length)

    if resample_to:
        data = np.array([scipy.signal.resample(x, resample_to) for x in data])
        
    if scale:
        data = min_max_scale(data)

    return data


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]
