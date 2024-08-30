from dataclasses import dataclass

import numpy as np

from .parameters import Parameters

import matplotlib.pyplot as plt


@dataclass
class ElectrodeSequence:
    electrodes: np.ndarray
    current_levels: np.ndarray
    phase_widths_us: float
    phase_gaps_us: float
    periods_us: float
    modes: int
    n_electrodes: int

    def plot(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        t = np.arange(self.electrodes.size) * (self.periods_us / 1e6)
        max_mag = max(self.current_levels) / 0.75
        line_h = self.electrodes - (self.current_levels / max_mag)

        z = np.ones(t.size) * np.nan
        y = np.c_[self.electrodes, line_h, z.reshape(-1, 1)].ravel()
        x = np.c_[t, t, z].ravel()

        ax.plot(x, y, color="black", linewidth=1)
        ax.set_ylabel("electrode")
        ax.set_xlabel("time [s]")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(1, n_electrodes))

        plt.grid()
        plt.show()

    def to_pulse_table(self) -> np.ndarray:
        data = np.zeros((self.n_electrodes, self.electrodes.size))

        for t, (electrode, amp) in enumerate(zip(self.electrodes, self.current_levels)):
            data[electrode - 1, t] = amp * 1e-4

        return data


def collate_into_sequence(
    channel_power: np.ndarray,
    parameters: Parameters,
):

    channel_order = np.arange(parameters.num_bands)
    if parameters.channel_order_type == "base_to_apex":
        channel_order = channel_order[::-1]

    num_bands, num_time_slots = channel_power.shape
    assert num_bands == parameters.num_bands

    channels = np.tile(channel_order, num_time_slots)
    magnitudes = channel_power[channel_order].ravel(order="F")

    finite_mag = np.isfinite(magnitudes)
    channels, magnitudes = channels[finite_mag], magnitudes[finite_mag]
    return channels, magnitudes


def channel_mapping(
    channels: np.ndarray,
    magnitudes: np.ndarray,
    parameters: Parameters,
) -> ElectrodeSequence:

    electrodes = parameters.electrodes[channels]

    volume = parameters.volume / 100
    ranges = parameters.upper_levels - parameters.lower_levels
    q_mag = np.clip(magnitudes / parameters.full_scale, None, 1).reshape(-1, 1)

    q_t = parameters.lower_levels[channels]
    q_r = ranges[channels]

    current_levels = (
        q_r * volume * q_mag
        if parameters.volume_type_standard
        else q_r * (q_mag + volume - 1)
    )

    current_levels = np.round(q_t + current_levels)
    current_levels[q_mag < 0] = 0
    return ElectrodeSequence(
        electrodes=electrodes,
        current_levels=current_levels,
        phase_widths_us=parameters.phase_width_us,
        phase_gaps_us=parameters.phase_gap_us,
        periods_us=parameters.period_us,
        modes=parameters.modes,
        n_electrodes=parameters.num_bands,
    )
