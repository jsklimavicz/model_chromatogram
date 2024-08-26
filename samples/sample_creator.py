from samples import Sample
from compounds import Compound, COMPOUND_LIBRARY


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

    def product_stability_samples(
        self, n_samples, timing_interval, compound_mapping
    ) -> list[Sample]:
        compound_id_set = set()
        for item in compound_mapping:
            compound_id_set.add(item["intial_compound"])
            assert len(item["compounds"]) == len(item["k"])
            for id in item["compounds"]:
                compound_id_set.add(id)
        print(compound_id_set)
