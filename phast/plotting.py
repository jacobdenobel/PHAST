import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_colors(sort: bool = True):
    if not sort:
        return list(mcolors.CSS4_COLORS.keys())
    colors = mcolors.CSS4_COLORS
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
        for name, color in colors.items()
    )
    return [name for _, name in by_hsv]

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
    # ax.set_xlabel("time")
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


def calculate_spike_rates(spikes, binsize, sound_duration, t_step):
    """Takes spikes as input with trials still separated and gives spike rates as vector"""
    num_trials = len(spikes)
    spike_rates = []
    if np.issubdtype(type(spikes[0][0]), int):
        spike_times = np.array(spikes, dtype=object) * t_step
    else:
        spike_times = spikes
    spike_times_vector = np.concatenate(spike_times).ravel()
    # sorted_spiketimes = np.sort(spike_times_vector)

    # calculate spike rates per bin
    for b, _ in enumerate(np.arange(binsize, sound_duration + binsize, binsize), 1):
        lower_lim = binsize * (b - 1)
        upper_lim = binsize * (b)
        spike_rates.append(
            ((spike_times_vector > lower_lim) & (spike_times_vector < upper_lim)).sum()
            / binsize
            / num_trials
        )

    return spike_rates

def PostSTH(ax, spikes, sound_duration, binsize, t_step):
    """Takes spikes as input with trials still separated"""
    num_bins = int(sound_duration / binsize)
    bins = np.linspace(0, sound_duration, num_bins)
    num_trials = len(spikes)
    spike_rates = calculate_spike_rates(spikes, binsize, sound_duration, t_step)
    # plot spike rates
    ax.plot(bins, spike_rates)
    ax.set_xlim(0, sound_duration)
    ax.xaxis.set_visible(False)


def PeriodSTH(ax, spike_rates, mod_freq, t_step, binsize, sound_duration):
    """Takes spike rates as input as a vector"""
    period = round(1 / mod_freq, 4)
    stepsize = int(round(period / binsize, 2))
    num_steps = np.floor(len(spike_rates) / stepsize)
    remainder = len(spike_rates) % stepsize
    # reshape into matrix of period per row
    spike_rates_per_period = np.reshape(
        spike_rates[: -1 - remainder + 1], (num_steps, stepsize)
    )
    # average over period
    mean_per_period = np.mean(spike_rates_per_period, 0)  # columnwise
    x_vector = np.linspace(0, period * 1000, stepsize)  # in [ms]
    # plot spike rates
    ax.bar(x_vector, mean_per_period)
    ax.set_xlim(0, period * 1000)
    ax.xaxis.set_visible(False)
    # ax.set_ylabel("Spikes/bin")

def PeriodSTH2(
    ax, spike_times, mod_freq, binsize, sound_duration, num_trials, start_modulation=0
):
    """Takes spike times as input as a vector"""
    ##
    period_duration = 1 / mod_freq
    end_bin = np.ceil(period_duration / binsize) * binsize + binsize

    mod_spike_timesn = spike_times[spike_times > start_modulation]
    p_bins = np.arange(start_modulation, 1, period_duration)
    mod_spike_timesn -= p_bins[np.digitize(mod_spike_timesn, p_bins, right=True) - 1]
    mod_spike_timesn = np.r_[0, mod_spike_timesn]  # SAME

    nn, edgesn = np.histogram(
        mod_spike_timesn * 1000, bins=np.arange(0, end_bin, binsize) * 1000
    )  # in ms
    ##

    period_duration = round(1 / mod_freq * 1000, 4)  # SAME
    binsize_ms = binsize * 1000  # [ms]
    end_bin = np.ceil(period_duration / binsize_ms) * binsize_ms + binsize_ms
    bins = np.arange(0, end_bin, binsize_ms)  # SAME as edges
    number_of_periods = int(
        np.floor(sound_duration / period_duration * 1000)
    )  # SAME as "(stop-start)/periode" in matlab
    start = start_modulation * 1000  # SAME as start
    mod_spike_times = [
        i * 1000 for i in spike_times if i > start_modulation
    ]  # [ms] SAME as B_here/PH
    # put everything in periods
    periods_collected = []  # SAME although hers began at 0 but wasn't correct
    # period_length = []
    for period_idx in range(0, number_of_periods):
        start_period = period_idx * period_duration + start  # SAME
        end_period = start_period + period_duration  # SAME
        PH_period = [i for i in mod_spike_times if start_period < i < end_period]
        PH_period -= start_period  # min and max values are approx the SAME
        # print('PH length', len(PH_period))
        # period_length.append(len(PH_period))
        periods_collected.extend(PH_period)  # SAME length
    # print('Mean PH length:', np.mean(period_length)) # averages are approx. the SAME
    periods_collected = np.r_[0, periods_collected]
    N_bins, edges = np.histogram(np.array([periods_collected]), bins=bins)  # [ms]
    ax.bar(
        edges[:-1], N_bins / num_trials / 0.2 * 1000 / number_of_periods
    )  # SAME edges
    # ax.hist(periods_collected, bins) # [ms]
    ax.set_xlim(0, period_duration)  # [ms]
    ax.xaxis.set_visible(False)

def calculate_ISI(spikes, t_step, sound_duration):
    """Takes spikes as input with trials still separated"""
    num_trials = len(spikes)
    rate_all = []
    IHI_all = []
    for trial in range(0, num_trials):
        spikes_per_trial = spikes[trial]
        rate_all.append(len(spikes_per_trial) / sound_duration)
        spike_times_per_trial = np.array(spikes_per_trial, dtype=object) * t_step
        delta_t = np.diff(spike_times_per_trial)
        IHI_all.extend(delta_t)

    return np.array(IHI_all), np.array(rate_all)

def ISIH(ax, spikes, mod_freq, sound_duration, binsize, t_step):
    """Takes spikes as input with trials still separated"""
    period_duration = round(1 / mod_freq, 4)
    num_trials = len(spikes)
    end_bin = np.ceil(period_duration * 10 / binsize) * binsize + binsize
    bins = np.arange(0, end_bin, binsize)
    ISI_all, rate_all = calculate_ISI(spikes, t_step, sound_duration)
    ISI_per_trial = np.asarray(ISI_all) / num_trials
    avg_rate = np.mean(rate_all)
    ax.hist(ISI_all * 1000, bins * 1000)  # [ms]
    ax.set_xlim(0, period_duration * 10 * 1000)  # [ms]

    return avg_rate

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
            x, data.mean(axis=0), 
            yerr=data.std(axis=0), 
            errorevery=100, ecolor="red"
        )
        ax.set_title(stat)
        ax.grid()
    plt.tight_layout()
    stochastic_threshold = np.vstack([fs.stochastic_threshold for fs in fiber_stats])
    axes[-1].hist(stochastic_threshold.ravel(), 1000)
    axes[-1].grid()

    axes[-1].set_title("stochastic threshold")
