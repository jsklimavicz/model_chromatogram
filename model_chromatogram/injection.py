from pydash import get as _get

from model_chromatogram.methods import InstrumentMethod, ProcessingMethod
from model_chromatogram.samples import Sample
from model_chromatogram.chromatogram import Baseline, PeakCreator
from model_chromatogram.system import System
import numpy as np
import datetime
from model_chromatogram.sequence import Sequence
from model_chromatogram.data_processing import PeakFinder
import uuid


class Injection:
    def __init__(
        self,
        sample: Sample,
        method: InstrumentMethod,
        processing_method: ProcessingMethod,
        sequence: Sequence,
        system: System,
        user: str | None = "admin",
        injection_time: datetime.datetime | None = None,
        init_setup=False,
    ) -> None:
        self.sample: Sample = sample
        self.user = user
        self.injection_uuid = str(uuid.uuid4())
        self.injection_time = (
            injection_time if injection_time is not None else datetime.datetime.now()
        )
        self.method: InstrumentMethod = method
        self.processing_method = processing_method
        self.system: System = system
        self.sequence = sequence
        self.init_setup = init_setup
        self.__add_to_sequence()
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

    def __add_to_sequence(self):
        self.sequence.add_injection(
            sample_name=self.sample.name,
            injection_time=self.injection_time,
            injection_uuid=self.injection_uuid,
        )

    def __calculate_compound_retention(self):
        """Iteratively calculates retention time for compounds.

        Currently implemented to use the column parameters, solvent profiles, solvent ph, and temperature to calculate
        the retention times of each compound.

        Returns:
            None
        """
        for compound in self.sample.compounds:
            compound.set_retention_time(
                column=self.system.column,
                solvent_profiles=self.method.profile_table,
                solvent_ph=self.method.ph,
                temperature=self.method.temperature,
                init_setup=self.init_setup,
            )

    def __create_chromatograms(self):
        """Creates initial background chromatograms based on solvent gradient.

        Returns:
            None
        """
        self.chromatograms: dict = {}
        # uv_channels
        for wavelength, name in zip(self.uv_wavelengths, self.uv_channel_names):
            times, signals = self.method.get_uv_background(wavelength)
            baseline = Baseline(
                np.array(times), np.array(signals), wavelength=wavelength
            )
            self.chromatograms[name] = baseline
        self.times = times.to_numpy()

    def __add_compounds(self):
        """Iteratively calculates signals for each peak.

        For each compound, a signal is created across the whole chromatogram. Then for each chromatogram, these
        original signals are multiple by each compound's absorbance at each chromatogram's wavelength, and then these
        signals are added to each chromatogram.

        Additionally, output son components are created after the chromatograms are produced.

        Returns:
            None
        """
        for compound in self.sample.compounds:
            compound_peak_signal = self.peak_creator.compound_peak(compound, self.times)
            compound_peak_signal /= self.method.dilution_factor
            compound_peak_signal *= self.method.injection_volume
            max_absorbance = compound.get_absorbance(self.uv_wavelengths)
            for name, absorbance in zip(self.uv_channel_names, max_absorbance):
                self.chromatograms[name].add_compound_peak(
                    absorbance=absorbance, signal=compound_peak_signal
                )

        for name, chromatogram in self.chromatograms.items():
            results_dict = {
                "channel_name": name,
                "peaks": [],
                "drift": chromatogram.signal[-1] - chromatogram.signal[0],
                "signal_noise": np.mean(chromatogram.signal[0:50]),
                "signal_statistic": {
                    "minimum": chromatogram.signal.min(),
                    "maximum": chromatogram.signal.max(),
                    "average": np.mean(chromatogram.signal),
                },
            }
            self.dict["results"].append(results_dict)
            datacube_dict = {
                "channel": name,
                "wavelength": chromatogram.wavelength,
                "times": chromatogram.times.tolist(),
                "times_unit": "MinuteTime",
                "signal": chromatogram.signal.tolist(),
                "signal_unit": "MilliAbsorbanceUnit",
            }
            self.dict["datacubes"].append(datacube_dict)

    def plot_chromatogram(self, channel_name, **kwargs):
        return self.chromatograms[channel_name].plot(**kwargs)

    def get_chromatogram_data(self, channel_name, **kwargs):
        return self.chromatograms[channel_name].get_chromatogram_data(**kwargs)

    def find_peaks(self, channel_name) -> PeakFinder:
        peak_finder = PeakFinder(
            *self.get_chromatogram_data(channel_name, pandas=False),
            processing_method=self.processing_method,
            sample_introduction=self.method.sample_introduction,
            channel_name=channel_name
        )
        for ind, result in enumerate(self.dict["results"]):
            if result["channel_name"] == channel_name:
                self.dict["results"][ind]["peaks"] = [
                    peak.get_properties() for peak in peak_finder
                ]

        return peak_finder

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, index):
        return self.chromatograms[index]

    def to_dict(self):
        seq_dict = self.sequence.lookup_injection(self.injection_uuid)
        runs_dict = {
            "runs": [
                {
                    "injection_time": self.injection_time.isoformat(),
                    "injection_number": _get(seq_dict, "injection_number"),
                    "injection_url": _get(seq_dict, "injection_url"),
                    "sequence": {
                        "name": _get(seq_dict, "name"),
                        "datavault": _get(seq_dict, "datavault"),
                        "url": _get(seq_dict, "url"),
                        "start_time": _get(seq_dict, "start_time"),
                        "last_update_time": _get(seq_dict, "last_update_time"),
                        "total_injections": _get(seq_dict, "total_injections"),
                    },
                }
            ]
        }
        return {**self.dict, **runs_dict}

    def _create_self_dict(self):
        self.dict = {
            "systems": [self.system.todict()],
            "users": [{"name": self.user}],
            "methods": [
                {
                    "injection": self.method.todict(),
                    "processing": self.processing_method.todict(),
                }
            ],
            "samples": [
                {
                    "name": self.sample.name,
                    "creation_date": self.sample.creation_date,
                    "location": self.sample.location,
                    "type": self.sample.type,
                }
            ],
            "results": [],
            "datacubes": [],
        }
