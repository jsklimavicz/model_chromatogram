from pydash import get as _get

from methods.method import Method
from compounds.sample import Sample
from chromatogram.chromatogram import Baseline
from chromatogram.peakcreator import PeakCreator


class Injection:
    def __init__(self, sample: Sample, method: Method) -> None:
        self.sample: Sample = sample
        self.method: Method = method
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
            max_absobances = compound.get_absorbance(self.uv_wavelengths)
            for name, absorbance in zip(self.uv_channel_names, max_absobances):
                self.chromatograms[name].add_compound_peak(
                    self.peak_creator, compound, absorbance
                )

    def plot_chromatogram(self, channel_name, **kwargs):
        self.chromatograms[channel_name].plot(**kwargs)
