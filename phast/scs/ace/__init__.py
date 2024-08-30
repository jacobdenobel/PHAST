"""

p = Ensure_field(p, 'map_name', 'ACE');
p = Ensure_amplitude_params(p);
p = Ensure_rate_params(p);

p = Append_process(p, @Audio_proc);
p = Append_process(p, @Freedom_microphone_proc);
p = Append_process(p, @FE_AGC_proc);

p = Ensure_field(p, 'envelope_method', 'vector sum');
switch p.envelope_method
case 'power sum'
	p = Append_process(p, @FFT_PS_filterbank_proc);
case 'vector sum'
	p = Append_process(p, @FFT_VS_filterbank_proc);
	p = Append_process(p, @Abs_proc);
otherwise
	error('envelope_method must be either "power sum" or "vector sum"');
end

p = Append_process(p, @Gain_proc);

if (p.channel_stim_rate_Hz ~= p.analysis_rate_Hz)
	p = Append_process(p, @Resample_FTM_proc);
end

p = Append_process(p, @Reject_smallest_proc);
p = Append_process(p, @LGF_proc);
p = Append_process(p, @Collate_into_sequence_proc)
"""

from .parameters import Parameters
from .audio import process_audio, freedom_mic
from .agc import agc
from .filterbank import filterbank, power_sum_envelope, envelope_method
from .utility import gain, resample, reject_smallest
from .lgf import lgf
from .mapping import collate_into_sequence, channel_mapping


def ace(wav_file: str, parameters: Parameters = None, **kwargs):
	if parameters is None:
 		parameters = Parameters(**kwargs)
   
	audio_signal, *_ = process_audio(wav_file, parameters)
	signal = freedom_mic(audio_signal, parameters)
	signal = agc(signal, parameters)
	spectrum = filterbank(signal, parameters)        

	channel_power = envelope_method(spectrum, parameters)
	channel_power = gain(channel_power, parameters)
	channel_power = resample(channel_power, parameters)
	channel_power = reject_smallest(channel_power, parameters)
	channel_power = lgf(channel_power, parameters)
	channels, magnitudes = collate_into_sequence(channel_power, parameters)
	electrode_seq = channel_mapping(channels, magnitudes, parameters)
	pulse_train = electrode_seq.to_pulse_table()

	return pulse_train, parameters, audio_signal
 
