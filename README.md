# Phenomological Adaptive STochastic auditory nerve fiber model

[![test-python](https://github.com/jacobdenobel/PHAST/actions/workflows/test.yml/badge.svg)](https://github.com/jacobdenobel/PHAST/actions/workflows/test.yml)

This repository contains an archive implementation of a phenomenological auditory nerve fiber model. The model is implemented in C++, with a runtime interface for Python. It can be easily installed via pip:
```bash
pip install phastc
```

The model has been tested and built for python versions 3.9-12, and is compiled using g++ for MacOS and Linux and msvc on Windows. 


## Running PHAST
The PHAST model simulates the response of a single auditory nerve fiber to electrical stimulation. As such, the model revolves arround two main inputs, the stimulus, and a list of fibers. These are both managed by objects in Python and explained in detail in the following sections.

### Pulse Train
The pulse train encodes the stimulus to be used in the simulation. We differentiate between a ```ConstantPulseTrain``` object and a ```PulseTrain``` object. The former can be used when the stimulation has a constant interval and amplitude, and can be a lot more efficient. The latter should be used whenever each pulse in the pulse train can be different, for example when using stimuli produced by a speech coding strategy. The ```PulseTrain``` has the following signature:

```python
stimulus = PulseTrain(
    pulse_train: np.ndarray = ...,
    time_step: float = 1e-6,
    ...
)
```
Only the first parameter is required, and should be a numpy array (matrix) of shape n electrodes x time steps. Then each element of the matrix encodes the amplitude of a pulse at given timestep for a given electrode. Ensure that the time steps of the matrix match the ```time_step`` parameter. 

The ```ConstantPulseTrain``` has the following signature:

```python
stimulus = ConstantPulseTrain(
    duration: float = ...,
    rate: float = ...,
    amplitude: float = ...,
    time_step: float = 1e-6,
)
```
Here, the ```duration``` denotes the total length in seconds of the pulse train, and ```rate``` the pulse rate, i.e. number of pulses per second. Each pulse in this pulse train has the same amplitude, specified by ```amplitude```. Again, ```time_step``` encodes the time step of the pulse train. Note that this object can only be used for single electrode stimulation. 


### Fiber
The ```Fiber``` object encodes a single fiber to be analyzed by the PHAST model. Many can be analyzed at the same time, so often a list of several fibers can be considered at any given time. The object has the following signature:

```python
fiber = Fiber(
    i_det: np.ndarray = ...,
    spatial_constant: np.ndarray = ...,
    sigma: np.ndarray = ...,
    fiber_id: int = ...,
    sigma_rs: float = ...,
    refractory_period: RefractoryPeriod = ...,
    decay: Decay = ...,
    store_stats: bool = False
)
```
Here ```i_det, spatial_constant, sigma```, are all vectors of length number of electrodes, which should match the number of electrodes in the ```PulseTrain``` used in stimulation. For ```i_det```, this defines the deteriministic threshold after which the fiber spikes, for a pulse from a given electrode. The ```spatial_constant``` defines an electrode specific spatial constant, which is used to scale stimulation. The ```sigma``` parameter is another electrode specific parameter, which is the relative spread per ```i_det```, i.e. ```relative_spread * i_det```. ```fiber_id``` encodes a unique identifier, specified by the used to attach to the fiber. ```sigma_rs``` is used for stochasticy between trials. ```store_stats``` defines whether all statistics should be stored, such as the refractoriness at each time step. This defaults to False, and should be used with caution, as this *significantly increases memory usage*.
```RefractoryPeriod``` is a parameter wrapper for handling both absolute and relative refractoriness. It can be defined as follows, and if not given explicitly to the ```Fiber```, the following default values are used:

```python
ref = RefractoryPeriod(
    absolute_refractory_period: float = 4e-4,
    relative_refractory_period: float = 8e-4,
    sigma_absolute_refractory_period: float = 0.0,
    sigma_relative_refractory_period: float = 0.0
)
```

#### Decay
Several different versions of the model can be used, which use a different model for controlling spike rate decay.
- __Exponential__: The first version of the model, as used in: __van Gendt, Margriet J., et al.__ *"A fast, stochastic, and adaptive model of auditory nerve responses to cochlear implant stimulation."* Hearing research 341 (2016): 130-143. This model uses an exponential decay function. 
- __Power Law__: In a subsequent paper, the exponential decay was replaced by a power law function, to better approximate long duratioin fiber behaviour, presented in: __van Gendt, Margriet J., et al.__ _"Short and long-term adaptation in the auditory nerve stimulated with high-rate electrical pulse trains are better described by a power law."_ Hearing Research 398 (2020): 108090.
- __Leaky Integrator__: In the latest paper, the power law decay was replaced by a leaky integrator, which was shown to approximate long duration fiber behavoir with comparable accuracy as the power law decay function, but using significantly less computational resources. This version should be prefered when performing large scale experiments with PHAST. The model was presented in: __de Nobel, Jacob, et al.__ _"Biophysics-inspired spike rate adaptation for computationally efficient phenomenological nerve modeling."_ Hearing Research 447 (2024): 109011.

To use any of the previously mentioned versions of the PHAST model, the ```decay``` parameter needs to be specific when constructing a ```Fiber``` object with the correct ```Decay``` object. For the __exponential__ decay model, we need to pass:
```python
decay = Exponential(
    adaptation_amplitude: float = 0.01,
    accommodation_amplitude: float = 0.0003,
    sigma_adaptation_amplitude: float = 0.0,
    sigma_accommodation_amplitude: float = 0.0,
    exponents: list[tuple] = [(0.6875, 0.088), (0.1981, 0.7), (0.0571, 5.564)],
)
```
For the power law model, the following is required:
```python
decay = Powerlaw(
    adaptation_amplitude: float = 2e-4,
    accommodation_amplitude: float = 8e-6,
    sigma_adaptation_amplitude: float = 0.0,
    sigma_accommodation_amplitude: float = 0.0,
    offset: float = 0.06,
    exp: float = -1.5
)
```
Finally, for the model which uses the leaky integrator, the following decay object needs to be passed:
```python
decay = LeakyIntegratorDecay(
    adaptation_amplitude: float = 7.142,
    accommodation_amplitude: float = 0.072,
    adaptation_rate: float = 0.014,
    accommodation_rate: float = 19.996
)
```

### phast
Then finally, we can combine the above to run the ```phast``` model (for a single fiber):
```python
fiber_stats = phast(
    [fiber],                        # A list of Fiber objects
    pt,                             # A PulseTrain object, either PulseTrain or ConstantPulseTrain
    n_jobs = -1,                    # The number of parallel cores to use (-1 is all available)
    n_trials = 1                    # The number of trials to generate for each fiber
    use_random = True               # Whether the experiment should use randomness
)
```
This yields a list of ```FiberStats``` objects which contains information about the experiment, such as the occurence of spikes (e.g.```fiber_stats[0].spikes```), or other statistics, when ```store_stats``` has been enabled. In order to get consitent results when enabling randomness i.e. when ```use_random = True```, a proper seed should be set, using ```phast.set_seed(seed_value)```, similar to how one would use ```np.random.seed```.

### Creating Neurograms
We include a helper to easily aggregate ```FiberStats``` data into neurograms in matrix form. To do this, the following can be used:
```python
neurogram = Neurogram(
    fiber_stats, 
    bin_size: float,    # The required binsize of the neurogram, every spike falling in the bin is summed
)
```
This then creates a neurogram matrix, which can be accessed via the ```data``` member, i.e. ```neurogram.data```.

We provide the following plotting utility to easiliy visualize these neurograms:
```python
fig, ax = plt.subplots()
plot_neurogram(
    neurogram,
    ax=ax,      # Optional
    fig=fig     # Optional
)
```

## Speech Coding Strategies
Note that we include several options for sound encoding and integrated with PHAST, see the folder [phast/scs](phast/scs/README.md) for more information. 


## Citation

```
@article{de2024biophysics,
title = {Biophysics-inspired spike rate adaptation for computationally efficient phenomenological nerve modeling},
journal = {Hearing Research},
volume = {447},
pages = {109011},
year = {2024},
issn = {0378-5955},
doi = {https://doi.org/10.1016/j.heares.2024.109011},
url = {https://www.sciencedirect.com/science/article/pii/S0378595524000649},
author = {Jacob {de Nobel} and Savine S.M. Martens and Jeroen J. Briaire and Thomas H.W. Bäck and Anna V. Kononova and Johan H.M. Frijns},
}
```