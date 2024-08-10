from compound import Compound


class Sample:
    def __init__(
        self,
        sample_name,
        location,
        comp_list,
        conc_list,
        num_random_peaks,
        max_random_conc,
    ) -> None:
        self.name = sample_name
        self.location = location
