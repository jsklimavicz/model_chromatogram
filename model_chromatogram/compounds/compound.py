import numpy as np
from pydash import get as _get
import pandas as pd
from model_chromatogram.compounds import UVSpectrum
from model_chromatogram.system import ColumnParameters, Column
import itertools


class Compound:
    def __init__(self, find_UV_spectrum=True, **kwargs) -> None:
        self.kwargs = kwargs
        self.name = _get(kwargs, "name").strip()
        self.id = _get(kwargs, "id")
        self.cas = _get(kwargs, "cas").strip()
        self.mw = float(kwargs["mw"])
        self.intrinsic_log_p = float(kwargs["logp"])
        self.asymmetry_addition = 0
        self.broadening_factor = 0
        self.h_donors: float = self.__set_initial_float("H_donor")
        self.h_acceptors: float = self.__set_initial_float("H_acceptor")
        self.refractivity: float = self.__set_initial_float("refractivity")
        self.log_s: float = self.__set_initial_float("log_s")
        self.tpsa: float = self.__set_initial_float("tpsa")
        self.pka_list: list[float] = self.__set_pK_list(_get(kwargs, "pka_list"))
        self.pkb_list: list[float] = self.__set_pK_list(_get(kwargs, "pkb_list"))
        if find_UV_spectrum:
            self.set_uv_spectrum()

    def __set_pK_list(self, vals: str) -> list[float]:
        if vals is None or vals == "":
            return []
        else:
            float_vals = []
            for val in vals.split(","):
                if val.strip() == "":
                    continue
                else:
                    float_vals.append(float(val))
            return float_vals

    def __copy__(self):
        cmpd_dict = self.kwargs.copy()
        cmpd_dict["find_UV_spectrum"] = False
        cmpd = self.__class__(**cmpd_dict)
        cmpd.spectrum = self.spectrum
        return cmpd

    def __deepcopy__(self):
        return self.__class__(**self.kwargs)

    def __set_initial_float(self, key, default=0):
        try:
            return float(_get(self.kwargs, key))
        except ValueError:
            return default

    def set_uv_spectrum(self):
        try:
            self.spectrum = UVSpectrum(self.cas)
        except Exception:
            # default UV spectrum
            self.spectrum = UVSpectrum("63-74-1")

    def get_absorbance(self, wavelength, pathlength=1):
        """ """
        absorbance = self.spectrum.get_epsilon(wavelength) * pathlength
        absorbance *= self.m_molarity / 1000
        return absorbance

    def set_concentration(self, concentration, unit):
        """
        Sets the concentration of the compound.

        Args:
            concentration (float): The numeric value of the concentration.
            creation_date (datetime): Date that the sample was created.
            unit (int): An int to signify which units are used for the concentration of the input concentrations.
                1: mg/ml (part per thousand w/v)
                2: ug/ml (ppm w/v)
                3: ng/ml (ppb w/v)
                4: umol/ml (or mM)
                5: nmol/ml (or uM)
        """
        if unit == 1:
            self.concentration = concentration * 1000  # mg/ml to ug/ml
            self.m_molarity = self.concentration / self.mw
        elif unit == 2:
            self.concentration = concentration  # already in ug/ml
            self.m_molarity = self.concentration / self.mw
        elif unit == 3:
            self.concentration = concentration / 1000  # ng/ml to ug/ml
            self.m_molarity = self.concentration / self.mw
        elif unit == 4:
            self.m_molarity = concentration  # already in mM
            self.concentration = self.m_molarity * self.mw
        elif unit == 5:
            self.m_molarity = concentration / 1000  # uM to mM
            self.concentration = self.m_molarity * self.mw

    def calculate_logD(self, pH_value: float):
        """Calculates logD (distribution coefficient) of the compound at the provided pH.

        Acidic and basic pka values are used to generate a list of microspecies, and calculate their proportions and
        charges. This information is then used to calculate the logD, average charge of the microspecies at the given
        pH, and a broadening factor for the peak that depends on the distribution of the microspecies.

        Args:
            pH_value (float): pH value at which to calculate the logD.

        """
        n_pka = len(self.pka_list)
        n_pkb = len(self.pkb_list)
        all_pka_values = [*self.pka_list, *self.pkb_list]
        n_groups = n_pka + n_pkb
        proportions = []
        a_states = list(itertools.product([0, -1], repeat=n_pka))
        b_states = list(itertools.product([1, 0], repeat=n_pkb))
        states = [list(fk + sk) for fk in a_states for sk in b_states]

        ratios = np.array([10 ** (pH_value - pKa) for pKa in all_pka_values])
        fractions = np.array([1 / (1 + ratio) for ratio in ratios])
        state_probabilities = []
        for state in states:
            prob = 1
            for i in range(n_groups):
                if (i < n_pka and state[i] == 0) or (i >= n_pka and state[i] == 1):
                    prob *= fractions[i]
                else:
                    prob *= 1 - fractions[i]

            state_probabilities.append(prob)
        proportions.append(state_probabilities)

        def calculate_logD(row, state_charges):
            weights = (
                np.where(
                    state_charges < 0,
                    -(state_charges**2) * 0.8,
                    -(state_charges**2) * 0.75,
                )
                + self.intrinsic_log_p
            )
            return np.dot(row, weights)

        state_charges = np.array([np.sum(state) for state in states])
        proportions = np.array(proportions)
        self.average_charge = np.dot(proportions, state_charges)[0]
        self.broadening_factor = 1 / np.sqrt(np.sum(proportions**2, axis=1))[0]
        self.logD = calculate_logD(proportions, state_charges)[0]

    def find_retention_factor(
        self,
        solvent_profiles: pd.DataFrame,
        solvent_ph: float,
        col_param: ColumnParameters,
    ) -> np.array:
        """Calculate a retention factor based on compound, solvent, and stationary phase parameters.

        Args:
            solvent_profiles (pd.DataFrame): DataFrame of solvent profiles with columns for `polarity`, `hb_acidity`,
            `hb_basicity`, and `dielectric`.
            solvent_ph (float): pH of the buffered solvent
            col_param (ColumnParameters): A `ColumnParameters` object containing attribute for the hydrophobic
            subtraction model.

        Returns:
            Rf (np.array): An array of retention factor values for each set of solvent profile points.

        """

        # polarity adjustments for solvent, column, and compound
        solv_p = solvent_profiles["polarity"].to_numpy()
        col_p = col_param.h
        self.calculate_logD(solvent_ph)
        Rf = (col_param.eb - 1) / 5
        Rf *= 2 + np.tanh((self.logD / 2 - 3) / (1 + np.exp(col_p * self.logD)))
        Rf *= np.exp((solv_p / 10) ** 1.5 + (1 + np.arcsinh(self.logD / 2 - 3)) / 2)

        def rational_func(x):
            return 2 / (1 + np.exp(-2 * x)) - 1

        # solvent/column H-bond acidity interaction with compound H-bond acceptors
        solv_a = solvent_profiles["hb_acidity"].to_numpy()
        col_a = col_param.a
        a = solv_a * self.h_acceptors**2 / (10 * col_a)
        # Rf *= np.tanh(a * np.exp(a) + 1)
        Rf *= rational_func(a * np.exp(a) + 1)

        # solvent/column H-bond basicity interaction with compound H-bond donors
        solv_b = solvent_profiles["hb_basicity"].to_numpy()
        col_b = col_param.b
        b = solv_b * self.h_donors**2 / (10 * col_b)
        # Rf *= np.tanh(b * np.exp(b) + 1)
        Rf *= rational_func(b * np.exp(b) + 1)

        # column interaction with size of compounds
        # Rf *= 2 + np.tanh(2 * col_param.s_star * self.mw ** (1 / 3))
        Rf *= 2 + rational_func(2 * col_param.s_star * self.mw ** (1 / 3))

        # solvent/column dielectric interaction with compound ratio of polar surface area
        solv_d = solvent_profiles["dielectric"].to_numpy()
        psa_v = self.tpsa**0.5 / self.mw ** (1 / 3)
        d = solv_d * psa_v / col_p
        # Rf *= np.tanh(d * np.exp(d) + 1)
        Rf *= rational_func(d * np.exp(d) + 1)

        Rf += 1  # add minimal retention factor of 1

        # add symmetric deviation depending on stationary phase retention of ions
        c7 = col_param.c7
        c28 = col_param.c28

        self.asymmetry_addition = self.average_charge * (
            c28 + (c7 - c28) / (7 - 2.8) * (solvent_ph - 2.8)
        )

        return Rf

    def set_retention_time(
        self,
        column: Column,
        solvent_profiles: pd.DataFrame,
        solvent_ph: float = 7.0,
        temperature=298,
        init_setup=False,
    ):
        """
        Calculates the retention time of a compound based on column volume, solvent properties, and compound properties.

        Args:
            column_volume (float): volume of column
            solvent_profiles (pd.DataFrame): dataframe containing corrsponding values for:
                time
                hydrogen bond acidity (hb_acidity)
                hydrogen bond basicity (hb_basicity)
                polarity
                dielectric
            solvent_ph (float): The pH of the solvent buffer.
            temperature (float): The temperature in Kelvin of the column during the injection.

        Returns:
            retention_time (float): Value of retention time.

        """

        Rf_0 = self.find_retention_factor(
            solvent_profiles, solvent_ph, column.parameters
        )
        Rf = Rf_0 * np.exp(
            (
                self.logD
                + np.sqrt(1 + self.h_acceptors**2 + self.h_donors**2)
                - 2 * self.log_s
            )
            * (10 * (1.0 / temperature - 1.0 / 298.0))
        )
        time = solvent_profiles["time"].to_numpy()
        flow = solvent_profiles["flow"].to_numpy() / column.volume

        move_ratio = np.cumsum(flow / Rf) * (time[1] - time[0]) - 1

        try:
            last_neg_ind = len(move_ratio[move_ratio < 0])
            end_index = last_neg_ind + 2
            switch_vals = move_ratio[last_neg_ind:end_index]
            switch_times = time[last_neg_ind:end_index]

            retention_time = switch_times[0] + (-switch_vals[0]) * (
                switch_times[1] - switch_times[0]
            ) / (switch_vals[1] - switch_vals[0])
        except IndexError:
            retention_time = time[-1] + 5

        self.retention_time = retention_time

        if init_setup:
            print(f"{self.cas}: \t {self.retention_time}")

        return retention_time
