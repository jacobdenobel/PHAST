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

from dataclasses import dataclass

from .parameters import AmplitudeParameters, RateParameters, Parameters
from .audio import process_audio, freedom_mic
from .agc import agc
from .filterbank import filterbank, power_sum_envelope
from .utility import gain, resample, reject_smallest
from .lgf import lgf
from .mapping import collate_into_sequence


def ace(wav_file: str):
    parameters = Parameters()
    audio, *_ = process_audio(wav_file, parameters)
    breakpoint()
