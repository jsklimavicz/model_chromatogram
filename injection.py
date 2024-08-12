from pydash import get as _get

from methods.method import Method
from compounds.sample import Sample
from compounds.compound import Compound
from chromatogram.chromatogram import Baseline
from chromatogram.peakcreator import PeakCreator
from system import System
import numpy as np


class Injection:
    def __init__(self, sample: Sample, method: Method, system: System) -> None:
        self.sample: Sample = sample
        self.method: Method = method
        self.system: System = system
        self.system.inject()
        self.peak_creator = PeakCreator()
        self.uv_wavelengths = []
        self.uv_channel_names = []
        for channel in _get(self.method.detection, "uv_vis_parameters"):
            self.uv_wavelengths.append(_get(channel, "wavelength"))
            self.uv_channel_names.append(_get(channel, "name"))
        self.__create_chromatograms()
        self.__add_compounds()

    def __create_chromatograms(self):
        self.chromatograms = {}
        # uv_channels
        for wavelength, name in zip(self.uv_wavelengths, self.uv_channel_names):
            times, signals = self.method.get_uv_background(wavelength)
            baseline = Baseline(times, signals)
            self.chromatograms[name] = baseline

    def __add_compounds(self):
        for compound in self.sample.compounds:

            # TODO: add method to modify default retention CV based on solvent and parameters
            # TODO: add method to convert cv to tr based on flow integral.
            compound.default_retention_time = self.__find_default_retention_time(
                compound
            )

            max_absobances = compound.get_absorbance(self.uv_wavelengths)
            for name, absorbance in zip(self.uv_channel_names, max_absobances):
                self.chromatograms[name].add_compound_peak(
                    self.peak_creator, compound, absorbance
                )

    def __find_default_retention_time(self, compound: Compound):
        # retention time is based on retention volume (column volume * retention in CV) and flow integral
        retention_volume = (
            compound.default_retention_CV * self.system.get_column_volume()
        )
        # get flow integral, and interpolate until we find the right volume.
        return np.interp(
            retention_volume,
            self.method.get_times(),
            self.method.get_profile("flow", order=1),
        )

    def plot_chromatogram(self, channel_name, **kwargs):
        self.chromatograms[channel_name].plot(**kwargs)
