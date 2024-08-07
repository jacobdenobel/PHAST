import numpy
from typing import overload

CAN_THREAD: bool
GENERATOR: RandomGenerator

class AbstractPulseTrain:
    n_used_electrodes: int
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def n_delta_t(self) -> int: ...
    @property
    def n_electrodes(self) -> int: ...
    @property
    def n_pulses(self) -> int: ...
    @property
    def n_unique_pulses(self) -> int: ...
    @property
    def sigma_ap(self) -> float: ...
    @property
    def steps_to_ap(self) -> int: ...
    @property
    def t_max(self) -> int: ...
    @property
    def time_step(self) -> float: ...
    @property
    def time_to_ap(self) -> float: ...

class ConstantPulseTrain(AbstractPulseTrain):
    def __init__(self, duration: float, rate: float, amplitude: float, time_step: float = ..., time_to_ap: float = ..., sigma_ap: float = ...) -> None: ...
    def get_pulse(self, arg0: int) -> Pulse: ...
    @property
    def amplitude(self) -> float: ...
    @property
    def pulse_interval(self) -> int: ...

class Decay:
    def __init__(self, *args, **kwargs) -> None: ...
    def compute_accommodation(self, arg0: int, arg1: FiberStats) -> float: ...
    def compute_adaptation(self, arg0: int, arg1: FiberStats, arg2: list[float]) -> float: ...
    def decay(self, arg0: int) -> float: ...
    def randomize(self, arg0: RandomGenerator) -> Decay: ...
    @property
    def time_step(self) -> float: ...

class Exponential(HistoricalDecay):
    def __init__(self, adaptation_amplitude: float = ..., accommodation_amplitude: float = ..., 
                 sigma_adaptation_amplitude: float = ..., sigma_accommodation_amplitude: float = ..., 
                 exponents: list[tuple[float, float]] = ...,
                 memory_size: int = ..., allow_precomputed_accommodation: bool = ..., cached_decay: bool = ..., cache: list[float] = ...) -> None: ...
    @property
    def exponents(self) -> list[tuple[float, float]]: ...

class Fiber:
    i_det: list[float]
    sigma: list[float]
    spatial_constant: list[float]
    def __init__(self, i_det: list[float], spatial_constant: list[float], sigma: list[float], fiber_id: int, n_max: int, sigma_rs: float = ..., refractory_period: RefractoryPeriod = ..., rel, decay: Decay = ..., store_stats: bool = ...) -> None: ...
    def process_pulse(self, arg0: Pulse, arg1: AbstractPulseTrain) -> None: ...
    def randomize(self) -> Fiber: ...
    @property
    def decay(self) -> Decay: ...
    @property
    def refractory_period(self) -> RefractoryPeriod: ...
    @property
    def stats(self) -> FiberStats: ...
    @property
    def threshold(self) -> float: ...

class FiberStats:
    def __init__(self, n_max: int = ..., fiber_id: int = ...) -> None: ...
    def __eq__(self, arg0: FiberStats) -> bool: ...
    @property
    def accommodation(self) -> numpy.ndarray: ...
    @property
    def adaptation(self) -> numpy.ndarray: ...
    @property
    def fiber_id(self) -> int: ...
    @property
    def n_pulses(self) -> int: ...
    @property
    def n_spikes(self) -> int: ...
    @property
    def pulse_times(self) -> numpy.ndarray: ...
    @property
    def refractoriness(self) -> numpy.ndarray: ...
    @property
    def scaled_i_given(self) -> numpy.ndarray: ...
    @property
    def spikes(self) -> numpy.ndarray: ...
    @property
    def stochastic_threshold(self) -> numpy.ndarray: ...
    @property
    def trial_id(self) -> int: ...

class HistoricalDecay(Decay):
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def accommodation_amplitude(self) -> float: ...
    @property
    def adaptation_amplitude(self) -> float: ...
    @property
    def memory_size(self) -> int: ...
    @property
    def sigma_accommodation_amplitude(self) -> float: ...
    @property
    def sigma_adaptation_amplitude(self) -> float: ...

class LeakyIntegrator:
    last_t: float
    rate: float
    scale: float
    value: float
    def __init__(self, scale: float = ..., rate: float = ...) -> None: ...
    def __call__(self, c: float, t: float) -> float: ...

class LeakyIntegratorDecay(Decay):
    def __init__(self, adaptation_amplitude: float = ..., accommodation_amplitude: float = ..., adaptation_rate: float = ..., accommodation_rate: float = ..., sigma_amp: float = ..., sigma_rate: float = ...) -> None: ...
    @property
    def accommodation(self) -> LeakyIntegrator: ...
    @property
    def adaptation(self) -> LeakyIntegrator: ...
    @property
    def sigma_amp(self) -> float: ...
    @property
    def sigma_rate(self) -> float: ...

class Period:
    mu: float
    def __init__(self, arg0: float) -> None: ...
    def tau(self, arg0: float) -> float: ...

class Point:
    x: float
    y: float
    def __init__(self, arg0: float, arg1: float) -> None: ...

class Powerlaw(HistoricalDecay):
    def __init__(self, 
                 adaptation_amplitude: float = ..., 
                 accommodation_amplitude: float = ..., 
                 sigma_adaptation_amplitude: float = ..., 
                 sigma_accommodation_amplitude: float = ..., 
                 offset: float = ..., 
                 exp: float = ...,
                 memory_size: int = ..., 
                 allow_precomputed_accommodation: bool = ..., 
                 cached_decay: bool = ..., cache: list[float] = ...) -> None: ...
    @property
    def exp(self) -> float: ...
    @property
    def offset(self) -> float: ...

class Pulse:
    def __init__(self, arg0: float, arg1: int, arg2: int) -> None: ...
    @property
    def amplitude(self) -> float: ...
    @property
    def electrode(self) -> int: ...
    @property
    def time(self) -> int: ...

class PulseTrain(AbstractPulseTrain):
    def __init__(self, pulse_train: list[list[float]], time_step: float = ..., time_to_ap: float = ..., sigma_ap: float = ...) -> None: ...
    def get_pulse(self, arg0: int) -> Pulse: ...
    @property
    def electrodes(self) -> numpy.ndarray: ...
    @property
    def pulse_times(self) -> numpy.ndarray: ...
    @property
    def pulses(self) -> numpy.ndarray: ...

class RandomGenerator:
    def __init__(self, seed: int, use_random: bool = ...) -> None: ...
    def __call__(self) -> float: ...

class RefractoryPeriod:
    def __init__(self, absolute_refractory_period: float = ..., relative_refractory_period: float = ..., sigma_absolute_refractory_period: float = ..., sigma_relative_refractory_period: float = ...) -> None: ...
    def compute(self, arg0: int, arg1: float, arg2, arg3) -> float: ...
    def randomize(self, arg0) -> RefractoryPeriod: ...
    @property
    def absolute(self) -> Period: ...
    @property
    def relative(self) -> Period: ...
    @property
    def sigma_absolute(self) -> float: ...
    @property
    def sigma_relative(self) -> float: ...

class WeightedExponentialSmoothing:
    expon: float
    n: int
    offset: float
    pla0: float
    prev_t: float
    scale: float
    tau: list[float]
    value: list[float]
    weight: list[float]
    def __init__(self, scale: float = ..., offset: float = ..., expon: float = ..., n: int = ...) -> None: ...
    def __call__(self, sample: float, time: float) -> float: ...

class WeightedExponentialSmoothingDecay(Decay):
    def __init__(self, adaptation_amplitude: float = ..., accommodation_amplitude: float = ..., sigma: float = ..., offset: float = ..., exp: float = ..., n: int = ...) -> None: ...
    @property
    def accommodation(self) -> WeightedExponentialSmoothing: ...
    @property
    def adaptation(self) -> WeightedExponentialSmoothing: ...
    @property
    def sigma(self) -> float: ...

def alpha_xy(arg0: float, arg1: float, arg2: float, arg3: float) -> float: ...
def get_alpha(arg0: float, arg1: float) -> float: ...
def get_tau(arg0: float, arg1: float) -> float: ...
def linspace(arg0: float, arg1: float, arg2: int) -> list[float]: ...
@overload
def phast(i_det: list[float], i_min: list[float], pulse_train: list[list[float]], decay: Decay, relative_spread: float = ..., n_trials: int = ..., refractory_period: RefractoryPeriod = ..., rel, use_random: bool = ..., fiber_id: int = ..., sigma_rs: float = ..., evaluate_in_parallel: bool = ..., time_step: float = ..., time_to_ap: float = ..., store_stats: bool = ...) -> list[FiberStats]: ...
@overload
def phast(fibers: list[Fiber], pulse_train: AbstractPulseTrain, evaluate_in_parallel: bool = ..., generate_trials: int = ..., use_random: bool = ...) -> list[FiberStats]: ...
def pla_at_perc(arg0: float, arg1: float, arg2: float) -> Point: ...
def pla_x(arg0: float, arg1: float, arg2: float) -> float: ...
def set_seed(seed: int) -> None: ...
