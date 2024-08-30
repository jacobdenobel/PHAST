# About

This module contains two different speech coding strategies for generating pulse trains for stimulating a CI, based on a given audio signal. This includes:
-   ACE (Cochlear)
-   SpecRes (Advanced Bionics)


## Usage
The simplest usage is through the respective end-to-end functions, which process a .wav file an produce a neurogram as output. For more fine-grained control, please take a look at the specifics for each strategy. The end to end strategy can be performed via the following for *ACE*:
```python
(audio_signal, sr), pulse_train, neurogram = phast.ace_e2e(
    phast.SOUNDS['asa'],
    ...
)
```
and via the following for SpecRes
```python
(audio_signal, sr), pulse_train, neurogram = phast.ab_e2e(
    phast.SOUNDS['asa'],
    ...
)
```
Both methods return the analyzed audio signal (```audio_signal```) and corresponding sample rate (```sr```). ```pulse_train``` is a n electrodes x n timesteps matrix of pulse ampltitudes, and ```neurogram``` is a ```Neurogram``` object, for which the ```neurogram.data``` member contains a n fibers x n timesteps matrix of spike counts. 

