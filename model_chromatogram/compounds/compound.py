import numpy as np
from pydash import get as _get
from model_chromatogram.compounds import UVSpectrum
from model_chromatogram.system import ColumnParameters, Column
import itertools
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import MolFromSmarts


class Compound:
    def __init__(self, find_UV_spectrum=True, **kwargs) -> None:
        self.kwargs = kwargs
        self.name = _get(kwargs, "name").strip()
        self.id = _get(kwargs, "id")
        try:
            self.smiles = _get(kwargs, "SMILES").strip()
            self.mol = MolFromSmarts(self.smiles)
            self.num_heavy_atoms = rdmd.CalcNumHeavyAtoms(self.mol)
        except AttributeError:
            self.smiles = None
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
        except (ValueError, TypeError):
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
        hb_acidity: np.array,
        hb_basicity: np.array,
        polarity: np.array,
        dielectric: np.array,
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

        self.calculate_logD(pH_value=7.0)
        vol_ratio = self.mw ** (1 / 3)
        ratio_tpsa = self.tpsa**0.5 / vol_ratio

        c7 = col_param.c7
        c28 = col_param.c28
        curr_c = c28 + (c7 - c28) / (7 - 2.8) * (solvent_ph - 2.8)

        # add symmetric deviation depending on stationary phase retention of ions
        self.asymmetry_addition = abs(self.average_charge) * (curr_c)

        log_rf = np.log(col_param.eb)

        # HB_Acidity Term
        a = np.sqrt(self.h_acceptors) * col_param.a / (1 + vol_ratio * hb_acidity)

        # HB_Basicity Term
        b = np.sqrt(self.h_donors) * col_param.b / (1 + vol_ratio * hb_basicity)

        # Polarity Term
        p = -self.logD * col_param.h / (10 + polarity)

        # Dielectric Term
        d = curr_c * ratio_tpsa / (1 + dielectric / 10)

        s = vol_ratio * col_param.s_star / 10

        log_rf -= 4 * (a + b + p + d + s)

        Rf = np.exp(log_rf) + 1

        return Rf

    def set_retention_time(
        self,
        column: Column,
        time: np.array,
        flow: np.array,
        hb_acidity: np.array,
        hb_basicity: np.array,
        polarity: np.array,
        dielectric: np.array,
        temperature: np.array,
        solvent_ph: float = 7.0,
        init_setup=False,
    ):
        """
        Calculates the retention time of a compound based on column volume, solvent properties, and compound properties.

        Args:
            column (Column): Column object containing parameters for the stationary phase.
            time (np.array): Array of time points for the solvent profile.
            flow (np.array): Array of flow rates for the solvent profile.
            hb_acidity (np.array): Array of hydrogen bond acidity values for the solvent profile.
            hb_basicity (np.array): Array of hydrogen bond basicity values for the solvent profile.
            polarity (np.array): Array of polarity values for the solvent profile.
            dielectric (np.array): Array of dielectric values for the solvent profile.
            solvent_ph (float): The pH of the solvent buffer.
            temperature (float): The temperature in Kelvin of the column during the injection.


        Returns:
            retention_time (float): Value of retention time.

        """

        Rf_0 = self.find_retention_factor(
            hb_acidity,
            hb_basicity,
            polarity,
            dielectric,
            solvent_ph,
            column.parameters,
        )
        Rf = Rf_0 * np.exp(
            # (
            #     self.logD
            #     # + np.sqrt(1 + self.h_acceptors**2 + self.h_donors**2)
            #     - 2 * self.log_s
            # )
            # *
            (10 * (1.0 / temperature - 1.0 / 298.0))
        )

        move_ratio = np.cumsum(flow / (Rf * column.volume)) * (time[1] - time[0]) - 1

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
