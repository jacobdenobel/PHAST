import os
from typing import List
from dataclasses import dataclass
from functools import cached_property
from enum import Enum

import numpy as np

from .constants import DATA_DIR
from .phastcpp import Fiber, RefractoryPeriod, LeakyIntegratorDecay


class FiberType(Enum):
    HEALTHY = 0
    SHORT_TERMINAL = 1
    NO_DENDRITE = 2


@dataclass
class ElectrodeConfiguration:
    m_level: np.ndarray
    t_level: np.ndarray

    insertion_angle: np.ndarray = None
    greenwood_f: np.ndarray = None
    position: np.ndarray = None
    pw: float = 18e-6
    alpha: np.ndarray = None

    @property
    def n_electrodes(self):
        return len(self.m_level)

    @property
    def cs_enabled(self):
        return self.alpha is not None

    @property
    def n_channels(self):
        if not self.cs_enabled:
            return self.n_electrodes

        return len(self.alpha) * (self.n_electrodes - 1)


@dataclass
class ThresholdProfile:
    i_det: np.ndarray
    electrode: ElectrodeConfiguration
    fiber_type: FiberType = FiberType.HEALTHY
    greenwood_f: np.ndarray = None
    position: np.ndarray = None
    angle: np.ndarray = None

    @property
    def n_fibers(self):
        return self.i_det.shape[0]

    @cached_property
    def i_min(self):
        return np.nanmin(self.i_det, axis=0)

    def spatial_factor(self, fiber_idx) -> np.ndarray:
        return self.i_min / self.i_det[fiber_idx, :]

    def sigma(self, fiber_idx, rs: float = 0.06) -> np.ndarray:
        return self.i_det[fiber_idx, :] * rs

    @staticmethod
    def from_idet(i_det: np.ndarray, pw: float = 18e-6) -> "ThresholdProfile":
        i_min = np.nanmin(i_det, axis=0)
        return ThresholdProfile(
            i_det, ElectrodeConfiguration(t_level=i_min, m_level=3 * i_min, pw=pw)
        )

    def create_fiberset(
        self,
        selected_fibers: np.ndarray = None,
        current_steering: bool = True,
        store_stats: bool = False,
        sigma_rs: float = 0.04,
        rs: float = 0.06,
        absolute_refractory_period: float = 4e-4,
        relative_refractory_period: float = 8e-4,
        sigma_absolute_refractory_period: float = 0.1e-3,
        sigma_relative_refractory_period: float = 0.5e-3,
        accommodation_amplitude: float = 0.072,
        adaptation_amplitude: float = 7.142,
        accommodation_rate: float = 0.014,
        adaptation_rate: float = 19.996,
        sigma_amp: float = 0.6e-2,
        sigma_rate: float = 0.6e-2,
        **kwargs
    ) -> List[Fiber]:

        if selected_fibers is None:
            selected_fibers = np.arange(self.n_fibers)

        picker = lambda x: x
        if not current_steering and self.electrode.cs_enabled:
            picker = lambda x: x[
                self.electrode.alpha.size // 2 :: self.electrode.alpha.size
            ]

        fibers = []
        for fiber_idx in selected_fibers:
            fibers.append(
                Fiber(
                    i_det=picker(self.i_det[fiber_idx]),
                    spatial_constant=picker(self.spatial_factor(fiber_idx)),
                    sigma=picker(self.sigma(fiber_idx, rs)),
                    sigma_rs=sigma_rs,
                    fiber_id=fiber_idx,
                    store_stats=store_stats,
                    refractory_period=RefractoryPeriod(
                        absolute_refractory_period,
                        relative_refractory_period,
                        sigma_absolute_refractory_period,
                        sigma_relative_refractory_period,
                    ),
                    decay=LeakyIntegratorDecay(
                        adaptation_amplitude,
                        accommodation_amplitude,
                        adaptation_rate,
                        accommodation_rate,
                        sigma_amp,
                        sigma_rate,
                    ),
                )
            )
        return fibers


def load_df120(ft: FiberType = FiberType.HEALTHY) -> "ThresholdProfile":
    """Elektrodes in datastructuur Df120 zijn klinisch genummerd, dus van apicaal (e=1) naar basaal (e=16)

    Df120(m) : Data voor morfologie m
               m=1 -> Gezonde vezels
               m=2 -> Short terminals
               m=3 -> Dendrietloze vezels

    Df120(m).T(e)  : T-level van elektrode e (monopolair gestimuleerd)
    Df120(m).M(e)  : M-level van elektrode e (monopolair gestimuleerd)

    Df120(m).alpha : Gebruikte waardes van de current steering parameter alpha; alpha=0 betekent monopolaire stimulatie op het apicale contact, alpha=1 op het basale

    Df120(m).Ae(e) : Insertiehoek van elektrode e (in graden vanaf het ronde venster)
    Df120(m).Fe(e) : Geschatte geluidsfrequentie elektrode e op basis van de Greenwood-functie (in kHz)
    Df120(m).Le(e) : Positie elektrode e gemeten in mm langs het basilair membraan (van basaal naar apicaal)

    Df120(m).An(f) : Cochleaire hoek van perifere uiteinde vezel f langs het basilair membraan (in graden vanaf het ronde venster)
    Df120(m).Ln(f) : Positie vezel f gemeten in mm langs het basilair membraan (van basaal naar apicaal)
    Df120(m).Fn(f) : Greenwood-frequentie vezel f (in kHz)

    Df120(m).TI_env_log2(ep,n,f) : Drempel van vezel f, gestimuleerd met elektrodepaar ep met alpha(n)
                                   Deze drempel is uitgedrukt in log2-eenheden van het input-bereik gegeven door hilbertEnvelopeFunc+noiseReductionFunc.

                                   Uit demo4_procedural van GMT:

                                   // sig_frm_hilbert    = hilbertEnvelopeFunc(par_hilbert, sig_frm_fft); % Hilbert envelopes
                                   // sig_frm_energy     = channelEnergyFunc(par_energy, sig_frm_fft, sig_smp_gainAgc); % channel energy estimates
                                   // sig_frm_gainNr     = noiseReductionFunc(par_nr, sig_frm_energy); % noise reduction
                                   // sig_frm_hilbertMod = sig_frm_hilbert + sig_frm_gainNr; % apply noise reduction gains to envelopes

                                   Hier geeft sig_frm_hilbertMod de input die in f120MappingFunc omgerekend wordt naar stroom-amplitudes op basis van de T+M-levels
                                   De eenheden van sig_frm_hilbertMod komen overeen met die van Df120(m).TI_env_log2

    Df120(m).TIa(ep,n,f)         : Stroom op apicale elektrode van elektrodepaar ep, bij alpha(n) op de drempel van vezel f (in mA)
    Df120(m).TIb(ep,n,f)         : Stroom op basale elektrode van elektrodepaar ep, bij alpha(n) op de drempel van vezel f (in mA)
    """

    fname = os.path.join(DATA_DIR, "df120.npy")
    data = np.load(fname, allow_pickle=True).item()
    elec = ElectrodeConfiguration(
        m_level=data["M"][ft.value] * 1e-3,
        t_level=data["T"][ft.value] * 1e-3,
        insertion_angle=data["Ae"][ft.value],
        greenwood_f=data["Fe"][ft.value] * 1e3,
        position=data["Le"][ft.value],
        alpha=data["alpha"][ft.value],
    )
    TIa = data["TIa"][ft.value] * 1e-3
    TIb = data["TIb"][ft.value] * 1e-3
    i_det = TIa + TIb
    i_det = np.nan_to_num(i_det, nan=np.nanmax(i_det, axis=0))
    i_det = np.flip(i_det[:, : i_det.shape[1], :].reshape(-1, i_det.shape[2]).T, axis=0)

    tp = ThresholdProfile(
        i_det=i_det,
        electrode=elec,
        angle=np.flip(data["An"][ft.value]),
        position=np.flip(data["Ln"][ft.value]),
        greenwood_f=np.flip(data["Fn"][ft.value] * 1e3),
        fiber_type=ft,
    )
    return tp
