"""PHAST auditory nerve fiber model."""

import os
import warnings
from collections.abc import Iterable
from time import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from .phastcpp import (
    Decay,
    HistoricalDecay,
    WeightedExponentialSmoothing,
    WeightedExponentialSmoothingDecay,
    LeakyIntegrator,
    LeakyIntegratorDecay,
    Exponential,
    Fiber,
    FiberStats,
    GENERATOR,
    CAN_THREAD,
    HistoricalDecay,
    Period,
    Powerlaw,
    Point,
    Pulse,
    AbstractPulseTrain,
    PulseTrain,
    ConstantPulseTrain,
    RandomGenerator,
    Period,
    RefractoryPeriod,
    phast,
    set_seed,
)

I_DET = os.path.join(os.path.realpath(os.path.dirname(__file__)), "idet.npy")


def generate_stimulus(
    duration: float = 0.4,
    amplitude: float = 0.001,
    rate: int = 5000,
    pw: float = 18e-6,
    time_step: float = None,
    pulse_duration: float = None,
    time_to_ap: float = 0,
    sigma_ap: float = 0,
    mod_depth: float = 0.1,
    mod_start: int = 0,
    mod_freq: float = 100,
    mod_method: str = "litvak2001",
    use_old_pulse_rate: bool = False,
    as_pulse_train: bool = True,
    n_channels: int = 1,
    index: int = 0,
) -> np.ndarray:
    """Generate a (modulated) stimulus.

    Parameters
    ----------
    duration: float = 0.4
        The duration of the stimulus in seconds.
    amplitude: float
        The amplitude of the stimulus in A.
    rate: int = 5000
        The pulse rate in pps.
    pw: float = 18e-6
        The phase width of each pulse.
    time_step: float = None
        The time step increment of the pulse train
    pulse_duration: float = None
        The total duration of the pulse, if not None, this overwrites pw
    time_to_ap: float = 0
        The time until an action potential is observed
    depth: float = 0.1
        Depth of modulation, denotes the relative size of the modulation w.r.t
        the input amplitude.
    mod_start: int = 0
        The index at which to start the modulation
    mod_freq: float = 100
        The frequency used for modulation
    mod_method: str = 'litvak2001'
        The type of modulation used.
    as_pulse_train: bool = False
        Return a PulseTrain object
    n_channels: int
        Passed to wrap_stimulus if as pulse_train == True
    index: int
        Passed to wrap_stimulus if as pulse_train == True
    Returns
    -------
    np.ndarray
        The generated stimulus.

    Notes
    -----
    For the figures from Hu models, the litvak2001 method is used for some reason.

    Pulse duration/width is only used to calculate the time step of the experiments
    """
    if pulse_duration is None:
        pulse_duration = 2 * pw

    pw = pulse_duration / 2

    mus = 1e-6
    if time_step is None:
        time_component = pulse_duration
        if time_to_ap != 0 and time_to_ap < pulse_duration:
            time_component = time_to_ap
        assert time_component > mus, "smallest time step this works for is 1 mu s"
        time_step = np.gcd(int(time_component / mus), int(1 / rate / mus)) * mus
    else:
        duration / time_step
        warnings.warn(
            "Using user-defined time steps, this can lead to rounding errors."
        )

    length = np.floor(duration / time_step).astype(int)
    pt = np.zeros(length)
    pulse_rate = np.floor(1 / rate / time_step).astype(int)

    if use_old_pulse_rate:  # using old pulse rate
        pulse_rate = int((pw / time_step) * np.floor(1 / rate / pw))

    pt[::pulse_rate] = amplitude

    if mod_depth > 0:
        xt = np.arange(length) * time_step  # np.linspace(0, duration, length)
        m_start = np.floor(length / duration * mod_start).astype(int)
        m_sin = np.sin(mod_freq * 2 * np.pi * xt[m_start:])
        if mod_method == "litvak2001":
            pt[m_start:] = (1 - mod_depth) * pt[m_start:] - mod_depth * (
                pt[m_start:] * m_sin
            )
        elif mod_method == "litvak2003a":
            A = pt[m_start:]
            t = xt[: len(A)]
            pt[m_start:] = A * (1 + mod_depth * np.sin(mod_freq * 2 * np.pi * t))
        elif mod_method == "hu":
            pt[m_start:] += mod_depth * pt[m_start:] * m_sin

    pt = wrap_stimulus(pt, n_channels, index)

    if as_pulse_train:
        pt = PulseTrain(pt, time_step, time_to_ap, sigma_ap)

    return pt


def wrap_stimulus(
    stimulus: np.ndarray,
    n_channels: int = 8,
    index: int = 7,
) -> np.ndarray:
    """Wrap a single stimulus in a pulse train of size n_channels at index.

    Parameters
    ----------
    n_channels: int
        The number of electrodes/channels in the pulse train
    indx: int
        The index at which to store the stimulus

    Returns
    -------
        np.ndarray
    """

    pulse_train = np.zeros((n_channels, stimulus.size))
    pulse_train[index, :] = stimulus
    return pulse_train


def spike_times(fiber_stats, time_step=18e-6):
    if not isinstance(fiber_stats, Iterable):
        return fiber_stats.spikes * time_step
    return np.hstack([fs.spikes * time_step for fs in fiber_stats])


def permute_spike_times(spike_times, time_step=18e-6):
    """Permute the spike rates, since they now are only extactly at the time steps
    of the model.

    Parameters
    ----------
    spike_times: np.ndarray
    time_step: float

    Returns
    -------
    np.ndarray
    """
    return np.maximum(np.random.normal(spike_times, time_step), spike_times)


def spike_rate(spike_times, num_bins=None, duration=0.4, binsize=0.05, n_trials=100):
    num_bins = num_bins or int(duration / binsize)
    counts, _ = np.histogram(spike_times, num_bins, (0, duration))
    return counts / n_trials / binsize


def isi(fiber_stats, time_step=18e-6, stack=True):
    def isi_(fs):
        return np.diff(fs.spikes) * time_step

    if isinstance(fiber_stats, Iterable):
        data = [isi_(fs) for fs in fiber_stats]
        if stack:
            data = np.hstack(data)
        return data

    return isi_(fiber_stats)


def plot_spikes(ax: plt.axes, spikes: list, xmax=None, pulse_width=None):
    m = 0
    if pulse_width != None:
        for i, fs in enumerate(spikes):
            ax.scatter(
                fs * pulse_width, np.ones(len(fs)) + i, c="black", s=1, alpha=0.8
            )
        m = pulse_width * (
            xmax + 1
        )  # +1 because of rounding it will not become full pulse train duration
        ax.set_xlim(0, m)
    else:
        for i, fs in enumerate(spikes):
            ax.scatter(fs, np.ones(len(fs)) + i, c="black", s=1, alpha=0.8)
            if any(fs):
                m = max(m, max(fs))
            ax.set_xlim(0, xmax or m)
    ax.set_xlabel("time")
    ax.set_ylabel("trial")


def post_stimulus_time_histogram(ax, spikes: list, pulse_width: float, bin_width=1e-3):
    try:
        n = int(max(map(max, spikes))) + 1
    except ValueError:
        return

    spike_history = np.zeros((len(spikes), n))
    for i, sp in enumerate(spikes):
        spike_history[i, sp] = 1

    sound_duration = pulse_width * (
        n
    )  # this may cause errors if future input is not of exact 0.3 /0.4s length (but instead 0.44s)
    num_bins = int(pulse_width * (n) / bin_width)
    _, spike_times_idx = np.nonzero(spike_history)
    spike_times = spike_times_idx * pulse_width
    bins = np.linspace(0, sound_duration, num_bins)
    ax.hist(
        spike_times, bins
    )  # opmaak: made histtype='stepfilled', rwidth = 2, facecolor = 'k', alpha = 1
    ax.set_xlim(
        0, pulse_width * (n)
    )  # +1 because of rounding it will not become full pulse train duration
    ax.set_ylabel(f"spikes/bin ({bin_width}s)")
    ax.grid()


def plot_pulse_train(pulse_train, figsize=(5, 2)):
    n_channels = min(pulse_train.shape)
    fig, (axes) = plt.subplots(n_channels, 1, sharex=True, sharey=True, figsize=figsize)
    colors = plt.cm.get_cmap("tab20")
    for e, (pulses, ax) in enumerate(zip(pulse_train, axes), 1):
        ax.plot(pulses, c=colors(e))
        ax.grid()
        ax.set_ylabel(e)
    ax.set_xlabel("time")
    plt.tight_layout()


def plot_fiber_stats(fiber_stats):
    if not any(fiber_stats):
        return

    stat_names = ["accommodation", "adaptation", "refractoriness"]
    _, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.ravel()
    x = fiber_stats[0].pulse_times

    for ax, stat in zip(axes, stat_names):
        data = np.vstack([getattr(fs, stat) for fs in fiber_stats])
        # data[np.isinf(data)] = np.max(data[~np.isinf(data)])
        ax.errorbar(
            x, data.mean(axis=0), yerr=data.std(axis=0), errorevery=100, ecolor="red"
        )
        ax.set_title(stat)
        ax.grid()
    plt.tight_layout()
    stochastic_threshold = np.vstack([fs.stochastic_threshold for fs in fiber_stats])
    axes[-1].hist(stochastic_threshold.ravel(), 1000)
    axes[-1].grid()

    axes[-1].set_title("stochastic threshold")
