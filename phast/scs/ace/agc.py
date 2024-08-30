import numpy as np

from .parameters import Parameters, from_dB


def agc(signal: np.ndarray, parameters: Parameters) -> np.ndarray:

    agc_attack_weight = 3.912 / (
        parameters.agc_attack_time_s * parameters.audio_sample_rate_Hz
    )
    agc_release_scaler = from_dB(
        -parameters.agc_step_dB
        / (parameters.agc_release_time_s * parameters.audio_sample_rate_Hz)
    )

    u = signal.copy()

    env_vec = np.zeros(u.shape)
    gain_vec = np.zeros(u.shape)
    raw_gain_vec = np.zeros(u.shape)

    agc_gain_state = 1.0
    agc_env_state = 0.0

    fwr = np.abs(u)
    for n in range(len(signal)):
        # Level dynamics:
        # - envelope peak tracking
        # - implements release time
        x = fwr[n]
        y = agc_env_state * agc_release_scaler
        agc_env_state = max(x, y)

        # Static compression:
        # - infinite compression
        raw_gain = (
            1.0
            if agc_env_state < parameters.agc_kneepoint
            else parameters.agc_kneepoint / agc_env_state
        )

        # Gain dynamics:
        # - smooth gain changes with a first order IIR Low Pass Filter,
        # - implements attack time
        agc_gain_state = (
            agc_attack_weight * raw_gain + (1 - agc_attack_weight) * agc_gain_state
        )

        # Save intermediate signals:
        env_vec[n] = agc_env_state
        gain_vec[n] = agc_gain_state
        raw_gain_vec[n] = raw_gain

    return u * gain_vec
