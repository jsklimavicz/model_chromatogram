from model_chromatogram.samples import Sample
from model_chromatogram.compounds import Compound, COMPOUND_LIBRARY

import numpy as np
from scipy.linalg import eig, inv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class SampleCreator:
    def __init__(self) -> None:
        pass

    def random_samples(
        self,
        n_samples,
        compound_id_list,
        concentration_bounds,
        n_random_compounds,
        random_concentrations,
        n_unknowns,
        unknown_concentrations,
    ) -> list[Sample]:
        pass

    def reaction_optimization_samples(
        self,
        n_samples,
        target_compound_id,
        secondary_compound_ids,
        n_unknowns,
        unknown_probability,
        secondary_compound_percentages,
        target_concentration,
    ) -> list[Sample]:
        pass

    def _create_reaction_matrix(self, compound_mapping, compound_names):
        n = len(compound_names)
        A = np.zeros((n, n))

        for reaction in compound_mapping:
            initial_idx = compound_names.index(reaction["initial_compound"])

            for i, products in enumerate(reaction["compounds"]):
                rate = reaction["k"][i]
                # Subtract from the initial compound
                A[initial_idx, initial_idx] -= rate

                # Add to the products
                if isinstance(products, list):
                    for product in products:
                        product_idx = compound_names.index(product)
                        A[product_idx, initial_idx] += rate
                else:
                    product_idx = compound_names.index(products)
                    A[product_idx, initial_idx] += rate

        return A

    def _solve_reaction_ODE(self, t_values, A, initial_concentrations):
        # Diagonalize the matrix A
        eigvals, eigvecs = eig(A)

        # Compute the matrix exponential for each time step
        C_t = []
        for t in t_values:
            exp_diag = np.diag(np.exp(eigvals * t))
            C_t.append(
                np.dot(
                    eigvecs,
                    np.dot(exp_diag, np.dot(inv(eigvecs), initial_concentrations)),
                )
            )

        return np.real(np.array(C_t))

    def _initial_kinetics_setup(
        self, time_points: np.array, compound_mapping: list[dict]
    ):
        compound_names = set()
        for reaction in compound_mapping:
            compound_names.add(reaction["initial_compound"])
            for products in reaction["compounds"]:
                if isinstance(products, list):
                    compound_names.update(products)
                else:
                    compound_names.add(products)
        compound_names = list(compound_names)

        initial_concentrations = np.zeros(len(compound_names))
        for reaction in compound_mapping:
            if "initial_concentration" in reaction.keys():
                initial_idx = compound_names.index(reaction["initial_compound"])
                initial_concentrations[initial_idx] = reaction["initial_concentration"]

        # Construct the matrix A
        A = self._create_reaction_matrix(compound_mapping, compound_names)

        # Solve the system using linear algebra
        concentrations_over_time = self._solve_reaction_ODE(
            time_points, A, initial_concentrations
        )

        return compound_names, concentrations_over_time

    def plot_kinetics(
        self,
        compound_mapping,
        start_day,
        end_day,
        n_points=1001,
        title=None,
        compound_name_mapping=None,
        **kwargs,
    ):
        times = np.linspace(start_day, end_day, n_points)
        compound_names, concentrations_over_time = self._initial_kinetics_setup(
            time_points=times, compound_mapping=compound_mapping
        )
        plt.figure(figsize=(10, 8))
        for name, conc_list in zip(compound_names, concentrations_over_time.T):
            if compound_name_mapping is not None:
                label = compound_name_mapping[name]
            else:
                label = name
            plt.plot(times, conc_list, label=label)

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        if title is not None:
            plt.title(f"{title}")
        plt.legend()
        plt.grid(True)

    def product_stability_samples(
        self,
        time_points: np.array,
        compound_mapping: list[dict],
        compound_name_mapping: dict = None,
        base_name: str = "test",
        start_date: datetime | None = None,
    ) -> list[Sample]:
        # Get the compound names and initial concentrations
        compound_names, concentrations_over_time = self._initial_kinetics_setup(
            time_points=time_points, compound_mapping=compound_mapping
        )
        sample_list: list[Sample] = []
        if compound_name_mapping:
            compound_aliases = [compound_name_mapping[name] for name in compound_names]
        else:
            compound_aliases = None
        for concentration_set, time in zip(concentrations_over_time, time_points):
            if start_date is not None:
                sample_time = start_date + timedelta(days=time)
            else:
                sample_time = datetime.now()
            sample = Sample(
                name=f"{base_name}_{time}",
                compound_id_list=compound_names,
                compound_concentration_list=concentration_set,
                compound_alias=compound_aliases,
                concentration_unit=4,
                creation_date=sample_time,
            )
            sample_list.append(sample)

        return sample_list
