"""PHAST auditory nerve fiber model."""

from dataclasses import dataclass
from re import T
from time import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from .utilities import *
from .plotting import *

from .phastcpp import (
    Decay, 
    WeightedExponentialSmoothing,
    WeightedExponentialSmoothingDecay,
    LeakyIntegrator,
    LeakyIntegratorDecay,
    Exponential, 
    Fiber, 
    FiberStats, 
    GENERATOR, 
    HistoricalDecay, 
    Period, 
    Powerlaw, 
    PulseTrain, 
    ConstantPulseTrain,
    RandomGenerator, 
    RefractoryPeriod,
    phast, 
    set_seed
)


class Constants:
    us: float = 1e-6
    time_step: float = 18 * us
    time_to_ap: float = 0 * us


class PythonPulseTrain:
    # precomputes a vector for decay over time during pulse train
    def __init__(
        self,
        pulse_train: np.ndarray,
        pla_offset: float = 0.06,
        pla_expon: float = -1.5,
        exponents: "list[tuple]" = [
            (0.6875, 0.088),
            (0.1981, 0.7),
            (0.0571, 5.564),
        ],  # (weight, exponent),
        use_pla: bool = True,
        time_step=Constants.time_step,
        time_to_ap = Constants.time_to_ap
    ):
        self.pla_offset = pla_offset
        self.pla_expon = pla_expon
        self.exponents = exponents
        self.time_step = time_step
        self.use_pla = use_pla

        self.n = max(pulse_train.shape)  # total number of samples

        self.pulse_times, self.electrodes = np.where(
            pulse_train.T > 0
        )  

        self.time_order = np.argsort(self.pulse_times)

        self.steps_to_ap = int(np.floor(time_to_ap / time_step))

        self.pulses = np.abs(pulse_train.T[pulse_train.T > 0])

        self.n_pulses = len(self.pulse_times)  # total number of pulses applied

        self.max_pla = pow(self.pla_offset, self.pla_expon)

        # Dit bepaald de tijd vanaf een pulse to een ap
        # is nu eigenlijk niet nodig omdat je maar 1 pulse shape hebt = 1 time to ap
        self.times_to_ap = np.ones(self.n_pulses) * time_to_ap

        self.decay0 = self._decay(0)
        self.decay_range = self._decay(0, True)

    def decay(self, time_step: int):
        return (self._decay(time_step) / self.decay0) * self.decay_range

    def _decay(self, time_step: int, force_exp: bool = False):
        """
        Parameters
        ----------
        time_step: int
            Dit is de orde van self.time_step 1us
        """

        if self.use_pla and not force_exp:
            return np.power(
                time_step * self.time_step + self.pla_offset, self.pla_expon
            )
        return np.sum(
            [
                weight * np.exp(-time_step * self.time_step / exponent)
                for (weight, exponent) in self.exponents  # decay = e^(t_i/tau_a)
            ],
            axis=0,
        )

    def __iter__(self):
        return zip(
            self.pulse_times[self.time_order],
            self.electrodes[self.time_order],
            self.pulses[self.time_order],
            self.times_to_ap[self.time_order],
        )


@dataclass
class PythonFiber:
    """
    i_det: np.ndarray  # [A], array of size: electrodes [16] x fibres [3200]
    spatial_constant: np.ndarray
    sigma: np.ndarray  # spread of relative spread? sigma = I_det * RS
    n_max: int
    adaptation_amplitude: float
    accommodation_amplitude: float
    absolute_refractory_period: float  # ARP
    relative_refractory_period: float  # RRP
    use_random: bool
    within_refractoriness_stochasticity: float = 0.05  # 5% of tau_ARP/tau_RRP, also in fraction in Matlab
    accommodation: float = 0
    adaptation: float = 0
    sigma_absolute_refractory_period: float = 0
    sigma_relative_refractory_period: float = 0
    sigma_rs: float = 0
    sigma_adaptation_amplitude: float = 0
    sigma_accommodation_amplitude: float = 0  # Dit bestaat niet bij margiet,
    memory_size: int = None

    """

    i_det: np.ndarray  # [A], array of size: electrodes [16] x fibres [3200]
    spatial_constant: np.ndarray
    sigma: np.ndarray  # spread of relative spread? sigma = I_det * RS
    n_max: int
    adaptation_amplitude: float
    accommodation_amplitude: float
    absolute_refractory_period: float  # ARP
    relative_refractory_period: float  # RRP
    use_random: bool
    within_refractoriness_stochasticity: float = (
        0.05  # 5% of tau_ARP/tau_RRP, also in fraction in Matlab
    )
    accommodation: float = 0
    adaptation: float = 0
    sigma_absolute_refractory_period: float = 0
    sigma_relative_refractory_period: float = 0
    sigma_rs: float = 0
    sigma_adaptation_amplitude: float = 0
    sigma_accommodation_amplitude: float = 0  # Dit bestaat niet bij margiet,
    memory_size: int = None  # this only applies to pulses not spikes

    def __post_init__(self):
        """If all sigmas are equal to zero when random is on, it's the same as Matlab file with all sigmas == 0"""
        self.rng = np.random.randn if self.use_random else np.zeros

        # Array to be filled with all spikes:
        # First row is for spike times in integer intervals of pulse_train.time_step
        # second for stimulating electrode at the time of the spikes,
        # Third is for the time in seconds of the spikes + time_to_ap
        self.spikes_fast = np.empty((3, self.n_max), dtype=int)

        # Array to be filled with all pulses
        # First column is time step in integer intervals of pulse_train.time_step
        # Second is the actual current of the pulse
        self.pulses_fast = np.empty((2, self.n_max), dtype=float)
        self.n_spikes = 0
        self.n_pulses = 0

        #### Inter-trial randomness ####
        self.sigma = self.i_det * max(
            0.0, (self.sigma[0] / self.i_det[0]) + (self.sigma_rs * self.rng(1)[0])
        )

        self.adaptation_amplitude = max(
            0.0,
            self.adaptation_amplitude
            + (self.sigma_adaptation_amplitude * self.rng(1)[0]),
        )
        self.accommodation_amplitude = max(
            0.0,
            self.accommodation_amplitude
            + (self.sigma_accommodation_amplitude * self.rng(1)[0]),
        )
        self.absolute_refractory_period = max(
            0.0,
            self.absolute_refractory_period
            + (self.sigma_absolute_refractory_period * self.rng(1)[0]),
        )
        self.relative_refractory_period = max(
            0.0,
            self.relative_refractory_period
            + (self.sigma_relative_refractory_period * self.rng(1)[0]),
        )

        self.acco_over_time = []
        self.adap_over_time = []
        ################################################################

    def compute_refractoriness(self, time_since_last_spike):
        if time_since_last_spike is None:
            return 1.0  # no refractoriness if no pulse has yet been applied
        # random number generator
        r1, r2 = self.rng(2)
        # add stochasticity to ARP: tau_temp_stoch_ARP = tau_ARP + sigma_ARP (= within_refr * tau_stoch_ARP) + rand(1)
        absolute_refractory_period = self.absolute_refractory_period + (
            self.within_refractoriness_stochasticity
            * self.absolute_refractory_period
            * r1
        )
        # add stochasticity to RRP: tau_stoch_RRP = tau_RRP + sigma_RRP + rand(1)
        relative_refractory_period = self.relative_refractory_period + (
            self.within_refractoriness_stochasticity
            * self.relative_refractory_period
            * r2
        )
        # no spiking possible in ARP
        if time_since_last_spike < absolute_refractory_period:
            return np.inf

        # calculate refractoriness in case of RRP:
        # R = 1 / (1 - e^ ( (-t+tau_ARP)/ tau_RRP) )
        return 1 / (
            1
            - np.exp(
                -(time_since_last_spike - absolute_refractory_period)
                / relative_refractory_period
            )
        )

    def process_pulse(self, t, e, i_given, time_to_ap, pulse_train: "PulseTrain"):
        time_since_last_spike: float = None
        time_since_spikes = t - self.spikes  # list of spikes indices
        if any(time_since_spikes):
            time_since_last_spike = time_since_spikes[-1] * pulse_train.time_step

        # add stochasticity to deterministic thresholds for all fibres
        stoch_threshold_i, *_ = self.i_det[e] + (self.sigma[e] * self.rng(1))

        refractoriness = self.compute_refractoriness(time_since_last_spike)

        # memory in time:
        memory = max(0, (t - self.memory_size)) if self.memory_size is not None else 0
        remembered_pulses = self.pulses[0, :] > memory
        remembered_spikes = self.spikes > memory

        self.adaptation = np.sum(
            self.adaptation_amplitude
            * self.i_det[self.electrodes[remembered_spikes]]
            * pulse_train.decay(time_since_spikes[remembered_spikes])
        )

        self.accommodation = np.sum(
            self.pulses[1, :][remembered_pulses]
            * pulse_train.decay((t - self.pulses[0, :][remembered_pulses].astype(int)))
        )

        threshold = (
            stoch_threshold_i * refractoriness + self.adaptation + self.accommodation
        )

        if i_given > threshold:
            # Only increase adaptation if there is a spike
            self.adaptation += self.adaptation_amplitude * self.i_det[e]

            self.spikes_fast[:, self.n_spikes] = (
                t + pulse_train.steps_to_ap,
                e,
                (t * pulse_train.time_step) + time_to_ap,
            )
            self.n_spikes += 1

        self.pulses_fast[:, self.n_pulses] = (
            t,
            self.accommodation_amplitude * i_given * self.spatial_constant[e],
        )
        self.n_pulses += 1
        
        self.acco_over_time.append(self.accommodation)
        self.adap_over_time.append(self.adaptation)

    @property
    def pulse_memory(self):
        if not self.memory_size or self.n_pulses < self.memory_size:
            return 0
        return self.n_pulses - self.memory_size

    @property
    def spikes(self):
        return self.spikes_fast[0, : self.n_spikes]

    @property
    def electrodes(self):
        return self.spikes_fast[1, : self.n_spikes]

    @property
    def pulses(self):
        return self.pulses_fast[:, : self.n_pulses]


# @timeit
# Dit model is gebaseerd op monopolar stimulation, moeten we niet ook bipolar proberen? Moet dit niet eigenlijk ook een variabele worden?
# RS is ook afhankelijk van PW, komt dit nu terug in het model? Dat is nu namelijk vast
# 3 verschillende SR-levels ook includeren, zoals werd geintroduceerd in zilany 2009?
# In Zilany 2018 zijn de absolute en de relative refractory period aan elkaar gecorreleerd, is dit iets wat wij mogelijk moeten overwegen?
# Model van Zilany is gemodelleerd aan de hand van dieren onder narcose, het effect van de stapediusreflex wordt zo niet opgenomen in het model toch?
# Accommodation en adaptation verhogen de threshold, maar facilitation verlaagt de threshold, een idee om dit ook toe te voegen?
# use neurogram similarity index measure (NSIM) for measuring similarity between the two models?
def phast_python(
    i_det: np.ndarray,  # [A] length = 16
    i_min: np.ndarray,  # [A] length = 16
    pulse_train: np.ndarray,
    relative_spread: float = 0.06,
    n_trials: int = 1,
    accommodation_amplitude=8e-6,
    adaptation_amplitude: float = 2e-4,
    absolute_refractory_period: float = 0.0004,  # [s]
    relative_refractory_period: float = 0.0008,  # [s]
    pla_offset: float = 0.06,
    pla_expon: float = -1.5,
    exponents: "list[tuple]" = [(1, 0.1)],
    use_power_law: bool = True,
    use_random: bool = True,
    sigma_absolute_refractory_period: float = 0,
    sigma_relative_refractory_period: float = 0,
    sigma_rs: float = 0,
    sigma_adaptation_amplitude: float = 0,
    evaluate_in_parallel: bool = False,
    time_step: float = Constants.time_step
):
    """
    TODO: Time step of 1us for decay and spike times
        - Sampling rate van SURROGATE: 5us -> pw is altijd een factor van 5us
    TODO: Variable time spike with time_to_action_potential
    """

    # create input, namely pulse train
    pulse_train = PythonPulseTrain(
        pulse_train, pla_offset, pla_expon, exponents, use_power_law, time_step
    )

    results = []
    for _ in range(n_trials):
        fiber = PythonFiber(
            i_det,
            i_min / i_det,  # spatial factor
            i_det * relative_spread,
            pulse_train.n_pulses,
            adaptation_amplitude,
            accommodation_amplitude,
            absolute_refractory_period,
            relative_refractory_period,
            use_random,
            sigma_absolute_refractory_period,
            sigma_relative_refractory_period,
            sigma_rs,
            sigma_adaptation_amplitude,
        )
        for t, electrode_i, i_given, time_to_ap in pulse_train:
            fiber.process_pulse(t, electrode_i, i_given, time_to_ap, pulse_train)

        results.append(fiber.spikes)

    return results
