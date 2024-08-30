import enum
from dataclasses import dataclass

import numpy as np


def to_dB(gain):
    return 20 * np.log10(np.abs(gain))


def from_dB(gain_dB):
    return np.power(10, gain_dB / 20.0)


def RMS_from_dB_SPL(spl, ref_db):
    return from_dB(spl - ref_db - to_dB(np.sqrt(2)))


class ImplantMode(enum.Enum):
    MP1 = -1
    MP2 = -2
    MP1_2 = -3


@dataclass
class Parameters:
    audio_dB_SPL: float = 65.0

    # A reference pure tone with amplitude +/-1.0 is defined to
    # represent a certain Sound Pressure Level:
    reference_dB_SPL: float = 95.0

    # These are the speech SPLs that give C-level and T-level stimulation:
    C_dB_SPL: float = 65.0
    T_dB_SPL: float = 25.0

    # Crest factor is the ratio between peak and RMS levels:
    speech_crest_factor_dB: float = 11

    # A reference pure tone will achieve C-level stimulation
    # at a lower RMS level than a speech signal:
    speech_bandwidth_factor_dB: float = 6

    # Envelope level that produces C-level stimulation:
    sat_level: float = 1.0

    # Derived parameters:
    agc_kneepoint: float = None
    gain_dB: float = None
    dynamic_range_dB: float = None

    # Implant Parameters
    chip: str = "CIC4"
    RF_clock_Hz: int = 5e6
    MIN_CURRENT_uA: float = 17.5
    CL0_uA: float = 0.0
    MAX_CURRENT_LEVEL: float = 255
    MAX_CURRENT_uA: float = 1750.0
    min_phase_width_us: float = 25.0
    max_phase_width_us: float = 400.0
    phase_width_us: float = 25.0
    phase_gap_us: float = 7.0
    MIN_SHORT_GAP_us: float = 12.0

    # Rate parameters
    audio_sample_rate_Hz: int = None
    channel_stim_rate_Hz: int = None
    analysis_rate_Hz: int = None
    block_shift: int = None

    electrodes: np.ndarray = None
    num_bands: int = None
    num_selected: int = None

    interval_length: int = None
    implant_stim_rate_Hz: int = None
    period_us: float = None

    # freedom mic
    mic_order: int = 128
    directivity: str = "dir"

    # Agc
    agc_attack_time_s: float = 0.005
    agc_release_time_s: float = 0.075
    agc_step_dB: int = 25

    # Filterbank
    block_length: int = 128
    # Added in proc
    window: np.ndarray = None
    bin_freq_Hz: np.ndarray = None

    # Envelope
    envelope_method: str = "power sum"
    equalise: bool = True

    # resample
    resample_method: str = "repeat"

    # lgf
    Q: float = 20
    sub_mag: float = -1e-10

    # collated into seq
    channel_order_type: str = "base_to_apex"

    # Channel mapping
    modes: ImplantMode = ImplantMode.MP1_2
    lower_levels: np.ndarray = None
    upper_levels: np.ndarray = None
    full_scale: float = 1.0
    volume: float = 100
    volume_type_standard: bool = True

    def __post_init__(self):
        # FE AGC calibration:
        # Speech at C-SPL should just reach the AGC kneepoint.
        # Firstly calculate the RMS level for this speech:
        speech_rms_C = RMS_from_dB_SPL(self.C_dB_SPL, self.reference_dB_SPL)

        # Then calculate the peak level for this speech:
        self.agc_kneepoint = from_dB(self.speech_crest_factor_dB) * speech_rms_C

        # LGF calibration:
        # Saturation in LGF is just reached for speech at C-SPL,
        # or for a reference tone that is a specified number of dB lower.
        tone_sat_level_dB_SPL = self.C_dB_SPL - self.speech_bandwidth_factor_dB

        # Calculate the RMS level for this reference tone:
        tone_rms_sat_level = RMS_from_dB_SPL(
            tone_sat_level_dB_SPL, self.reference_dB_SPL
        )

        # Calculate the peak level for this reference tone:
        tone_peak_sat_level = tone_rms_sat_level * np.sqrt(2)

        # The filterbank has unity gain, so for a pure tone,
        # the envelope amplitude equals the peak value of the tone.
        # Calculate the gain needed to amplify this reference tone to sat level:
        self.gain_dB = to_dB(self.sat_level / tone_peak_sat_level)

        self.dynamic_range_dB = self.C_dB_SPL - self.T_dB_SPL

        self.CURRENT_BASE = self.MAX_CURRENT_uA / self.MIN_CURRENT_uA

        self.audio_sample_rate_Hz = self.RF_clock_Hz / 320

        self.channel_stim_rate_Hz = 1000
        self.analysis_rate_Hz = self.channel_stim_rate_Hz

        # Quantise_analysis_rate
        self.block_shift = np.ceil(self.audio_sample_rate_Hz / self.analysis_rate_Hz)
        self.analysis_rate_Hz = self.audio_sample_rate_Hz / self.block_shift
        self.channel_stim_rate_Hz = self.analysis_rate_Hz

        # Ensure electrodes
        self.electrodes = np.arange(1, 23)[::-1].reshape(-1, 1)
        self.num_bands = len(self.electrodes)
        self.num_selected = min(self.num_bands, 12)

        self.interval_length = np.round(
            self.analysis_rate_Hz / self.channel_stim_rate_Hz
        )
        self.implant_stim_rate_Hz = self.channel_stim_rate_Hz * self.num_selected

        period_tk = np.round(self.RF_clock_Hz / self.implant_stim_rate_Hz)
        self.period_us = 1e6 * period_tk / self.RF_clock_Hz

        if self.lower_levels is None:
            self.lower_levels = np.ones((self.num_bands, 1))
            self.upper_levels = 100 * np.ones((self.num_bands, 1))
        else:
            self.lower_levels = self.lower_levels.reshape(self.num_bands, 1)
            self.upper_levels = self.upper_levels.reshape(self.num_bands, 1)
