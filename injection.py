from pydash import get as _get

from methods import InstrumentMethod
from samples import Sample
from chromatogram import Chromatogram, Baseline, PeakCreator
from system import System
import numpy as np


class Injection:
    def __init__(
        self, sample: Sample, method: InstrumentMethod, system: System
    ) -> None:
        self.sample: Sample = sample
        self.method: InstrumentMethod = method
        self.system: System = system
        self.system.inject()
        self.peak_creator = PeakCreator(system=self.system)
        self.uv_wavelengths = []
        self.uv_channel_names = []
        for channel in _get(self.method.detection, "uv_vis_parameters"):
            self.uv_wavelengths.append(_get(channel, "wavelength"))
            self.uv_channel_names.append(_get(channel, "name"))
        self.__calculate_compound_retention()
        self.__create_chromatograms()
        self.__add_compounds()

    def __calculate_compound_retention(self):
        for compound in self.sample.compounds:
            compound.set_retention_time(
                self.system.get_column_volume(), self.method.profile_table
            )

    def __create_chromatograms(self):
        self.chromatograms: dict[Chromatogram] = {}
        # uv_channels
        for wavelength, name in zip(self.uv_wavelengths, self.uv_channel_names):
            times, signals = self.method.get_uv_background(wavelength)
            baseline = Baseline(np.array(times), np.array(signals))
            self.chromatograms[name] = baseline
        self.times = times

    def __add_compounds(self):
        for compound in self.sample.compounds:
            compound_peak_signal = self.peak_creator.compound_peak(compound, self.times)
            max_absobances = compound.get_absorbance(self.uv_wavelengths)
            for name, absorbance in zip(self.uv_channel_names, max_absobances):
                self.chromatograms[name].add_compound_peak(
                    absorbance=absorbance, signal=compound_peak_signal
                )

    def plot_chromatogram(self, channel_name, **kwargs):
        return self.chromatograms[channel_name].plot(**kwargs)

    def get_chromatogram_data(self, channel_name, **kwargs):
        return self.chromatograms[channel_name].get_chromatogram_data(**kwargs)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, index):
        return self.chromatograms[index]
