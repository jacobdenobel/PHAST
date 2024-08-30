import numpy as np

from .parameters import Parameters, from_dB


def lgf_alpha(Q, base_level, sat_level):
    if Q == 20 and base_level == 0.01 and sat_level == 1:
        return 340.8338
    raise NotImplementedError()


def lgf(
    channel_power: np.ndarray,
    parameters: Parameters,
) -> np.ndarray:

    base_level = parameters.sat_level / from_dB(parameters.dynamic_range_dB)
    alpha = lgf_alpha(parameters.Q, base_level, parameters.sat_level)

    # Scale the input between base_level and sat_level:
    # Find all the inputs that are above sat_level (i.e. r > 1)
    r = np.clip(
        (channel_power - base_level) / (parameters.sat_level - base_level), None, 1
    )
    # Find all the inputs that are below base_level (i.e. r < 0)
    sub = r < 0
    r[sub] = 0

    # Logarithmic compression:
    v = np.log(1 + (alpha * r)) / np.log(1 + alpha)
    v[sub] = parameters.sub_mag
    return v
