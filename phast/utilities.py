"""Standalone functions and utilities."""

import warnings
from collections.abc import Iterable
from time import time
from functools import wraps

import numpy as np
from .phastcpp import PulseTrain


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
    index: int = 0
) -> np.ndarray:
    """Generate a (modulated) cathodic/anodic biphasic stimulus.

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
        warnings.warn("Using user-defined time steps, this can lead to rounding errors.")
    
    length = np.floor(duration / time_step).astype(int)
    pt = np.zeros(length)
    pulse_rate = np.floor(1 / rate / time_step).astype(int)

    if use_old_pulse_rate: # using old pulse rate
        pulse_rate = int((pw / time_step) * np.floor(1 / rate / pw))
    
    pt[::pulse_rate] = amplitude
        
    if mod_depth > 0:
        xt = np.arange(length) * time_step #np.linspace(0, duration, length)
        m_start = np.floor(length / duration * mod_start).astype(int)
        m_sin = np.sin(mod_freq * 2 * np.pi * xt[m_start:])
        if mod_method == "litvak2001":
            pt[m_start:] = (1 - mod_depth) * pt[m_start:] - mod_depth * (pt[m_start:] * m_sin)
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

def box_muller(size, mu=0, sigma=1):
    """Method to generator Gaussian random numbers from a uniform distribution.
    Uses np.random.rand to generate the uniform random numbers required
    to compute the box-fuller transform.

    Parameters
    ----------
    size: tuple[int]
        size of the output array
    mu: float
        mean of the Gaussian distribution
    sigma: float
        std. dev. of the Gaussian distribution

    Returns
    -------
    np.ndarray[size]
        Gaussian random numbers
    """
    n = np.prod(size)
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    mag = sigma * np.sqrt(-2 * np.log(u1))
    r1 = mag * np.cos(2 * np.pi * u2) + mu
    r2 = mag * np.sin(2 * np.pi * u2) + mu
    return np.reshape(np.c_[r1, r2].ravel()[:n], size, "F")


def timeit(f):
    @wraps(f)
    def inner(*a, **k):
        t = time()
        r = f(*a, **k)
        print(f"{f.__qualname__} time elapsed: {time() - t} s")
        return r

    return inner


def nrmse(x, y):
    return np.sqrt(np.mean(np.power(x - y, 2))) / (x.max() - x.min())


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


def add_jitter(spikes, time_step, pulse_width):
    """Jitter is the standard deviation of a sampling of spike latencies.
    According to Miller (https://doi.org/10.1016/S0378-5955(99)00012-X)
    mean latency and jitter depend on stimulus level"""
    mean_latency = 0.7e-3
    SD_jitter = 0.0708e-3  # line 159 from PHAST_core.m
    jittered_spike_times = []
    for trial in range(0, len(spikes)):
        spike_times = (
            spikes[trial] * time_step
        )  # IS THIS CORRECT NOW WITH THE NEW TIME STEP FORMAT?
        latency = np.random.normal(mean_latency, SD_jitter, len(spike_times))
        latency[latency > pulse_width] = pulse_width
        jittered_spike_times.append(
            np.asarray(spike_times) + latency
        )  # currently only postponing spikes never earlier

    return jittered_spike_times
