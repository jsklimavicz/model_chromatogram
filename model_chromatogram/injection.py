from pydash import get as get_, set_

from model_chromatogram.methods import InstrumentMethod, ProcessingMethod
from model_chromatogram.samples import Sample
from model_chromatogram.chromatogram import Baseline, PeakCreator
from model_chromatogram.system import System
import numpy as np
import datetime
from model_chromatogram.sequence import Sequence
from model_chromatogram.data_processing import PeakFinder
import uuid
from model_chromatogram.user_parameters import BASELINE_NOISE


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
        for channel in self.method.detection:
            self.uv_wavelengths.append(get_(channel, "wavelength"))
            self.uv_channel_names.append(get_(channel, "name"))
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

        for channel in self.method.detection:
            if channel["detector_name"].lower() in ["uv", "pda", "fld", "mwd", "vwd"]:
                times, signals = self.method.get_uv_background(
                    get_(channel, "wavelength")
                )
                noise_level = BASELINE_NOISE
            else:
                times, signals = self.method.get_zero_background()
                noise_level = BASELINE_NOISE / 2
            baseline = Baseline(
                np.array(times),
                np.array(signals),
                channel_settings=channel,
                noise_level=noise_level,
            )
            self.chromatograms[get_(channel, "name")] = baseline
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
                print(name, absorbance, type(absorbance), str(absorbance))
                if absorbance is not None and str(absorbance) != "nan":
                    self.chromatograms[name].add_compound_peak(
                        absorbance=absorbance, signal=compound_peak_signal
                    )

        chromatograms = []
        for name, chromatogram in self.chromatograms.items():

            if name.lower() in ["temp", "temperature"]:
                chromatogram.signal += self.method.profile_table["temperature"].values

            if name.lower() in ["pressure"]:
                chromatogram.signal += self.method.profile_table["pressure"].values

            results_dict = {
                "channel_name": name,
                "fk_chromatogram": chromatogram.uuid,
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
                "fk_chromatogram": chromatogram.uuid,
                "wavelength": get_(
                    chromatogram.detection_settings, "wavelength", default=None
                ),
                "times": chromatogram.times.tolist(),
                "times_unit": "MinuteTime",
                "signal": chromatogram.signal.tolist(),
                "signal_unit": get_(
                    chromatogram.detection_settings, "unit", default=None
                ),
            }
            self.dict["datacubes"].append(datacube_dict)

            chromatogram_dict = {
                "pk": chromatogram.uuid,
                "channel_name": name,
                "detector_device": get_(
                    chromatogram.detection_settings, "detector_name"
                ),
                "fk_module": get_(chromatogram.detection_settings, "fk_module"),
            }
            chromatograms.append(chromatogram_dict)
        set_(self.dict, "methods.0.chromatograms", chromatograms)

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
                    "injection_number": get_(seq_dict, "injection_number"),
                    "injection_url": get_(seq_dict, "injection_url"),
                    "sequence": {
                        "name": get_(seq_dict, "name"),
                        "datavault": get_(seq_dict, "datavault"),
                        "url": get_(seq_dict, "url"),
                        "start_time": get_(seq_dict, "start_time"),
                        "last_update_time": get_(seq_dict, "last_update_time"),
                        "total_injections": get_(seq_dict, "total_injections"),
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
                    "chromatograms": [],
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
