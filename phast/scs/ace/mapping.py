import numpy as np

from .parameters import Parameters

def collate_into_sequence(
    channel_power: np.ndarray,
    parameters: Parameters,
    channel_order_type: str = 'base_to_apex'
):
    
    channel_order = np.arange(parameters.rate.num_bands)
    
    if channel_order_type == 'base_to_apex':
        channel_order = channel_order[::-1]
    

    [num_bands, num_time_slots] = channel_power.shape
    assert num_bands == parameters.rate.num_bands

    channels = np.tile(channel_order, num_time_slots)
    magnitudes = channel_power[channel_order].ravel()
    
    finite_mag = np.isfinite(magnitudes)
    return channels[finite_mag], magnitudes[finite_mag]
    

