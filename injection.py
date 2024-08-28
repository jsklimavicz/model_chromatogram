from pydash import get as _get
from pydash import set_

from methods import InstrumentMethod, ProcessingMethod
from samples import Sample
from chromatogram import Chromatogram, Baseline, PeakCreator
from system import System
import numpy as np
import datetime

from data_processing import PeakFinder


class Injection:
    def __init__(
        self,
        sample: Sample,
        method: InstrumentMethod,
        processing_method: ProcessingMethod,
        system: System,
        user: str | None = None,
        injection_time: datetime.datetime | None = None,
    ) -> None:
        self.sample: Sample = sample
        self.user = user
        self.injection_time = injection_time
        self.method: InstrumentMethod = method
        self.processing_method = processing_method
        self.system: System = system
        self.system.inject()
        self.peak_creator = PeakCreator(system=self.system)
        self._create_self_dict()
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

        for name, chromatogram in self.chromatograms.items():
            results_dict = {"channel_name": name, "peaks": []}
            self.dict["results"].append(results_dict)
            datacube_dict = {
                "channel": name,
                "times": chromatogram.times.tolist(),
                "times_unit": "MinuteTime",
                "signal": chromatogram.signal.tolist(),
                "signal_unit": "MilliAbsorbanceUnit",
            }
            self.dict["datacubes"].append(datacube_dict)

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

    def find_peaks(self, channel_name):
        peak_finder = PeakFinder(
            *self.get_chromatogram_data(channel_name, pandas=False),
            processing_method=self.processing_method
        )
        for ind, result in enumerate(self.dict["results"]):
            if result["channel_name"] == channel_name:
                self.dict["results"][ind]["peaks"] = [
                    peak.get_properties() for peak in peak_finder
                ]

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, index):
        return self.chromatograms[index]

    def to_dict(self):
        return self.dict

    def _create_self_dict(self):
        self.dict = {
            "systems": [self.system.todict()],
            "users": [{"name": self.user}],
            "runs": [{"injection_time": self.injection_time}],
            "methods": [
                {
                    "injection": self.method.todict(),
                    "processing": self.processing_method.todict(),
                }
            ],
            "samples": [
                {"name": self.sample.name, "creation_date": self.sample.creation_date}
            ],
            "results": [],
            "datacubes": [],
        }
