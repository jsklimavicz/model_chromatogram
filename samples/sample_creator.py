from samples import Sample
from compounds import Compound, COMPOUND_LIBRARY

import numpy as np
from scipy.linalg import eig, inv


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

    def product_stability_samples(
        self, time_points: np.array, compound_mapping: list[dict]
    ) -> list[Sample]:
        # Get the compound names and initial concentrations
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
        print(concentrations_over_time)
