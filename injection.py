from pydash import get as _get

from methods.method import Method
from compounds.sample import Sample
from chromatogram.chromatogram import Baseline
from chromatogram.peakcreator import PeakCreator


class Injection:
    def __init__(self, sample: Sample, method: Method) -> None:
        self.sample: Sample = sample
        self.method: Method = method
        self.__create_chromatograms()

    def __create_chromatograms(self):
        self.chromatograms = {}
        for channel in _get(self.method.detection, "uv_vis_parameters"):
            wavelength = _get(channel, "wavelength")
            times, signals = self.method.get_uv_background(wavelength)
            baseline = Baseline(times, signals)
            baseline.plot()

    def __add_compounds(self):
        peak_creator = PeakCreator()
        for compound in self.sample.compounds:
            pass
