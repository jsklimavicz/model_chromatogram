from pydash import get as _get
from model_chromatogram.compounds import Compound


class Solvent(Compound):
    def __init__(self, **kwargs) -> None:
        if "id" not in kwargs:
            id = [a.strip() for a in kwargs["id_list"].split(",")]
            kwargs["id"] = id
        if "logp" not in kwargs:
            kwargs["logp"] = 0
        if "default_CV" not in kwargs:
            kwargs["default_CV"] = 0
        if "asymmetry_addition" not in kwargs:
            kwargs["asymmetry_addition"] = 0
        super().__init__(**kwargs)
        self.hb_acidity = float(kwargs["hb_acidity"])
        self.hb_basicity = float(kwargs["hb_basicity"])
        self.dipolarity = float(kwargs["dipolarity"])
        self.polarity = float(kwargs["polarity"])
        self.dielectric = float(kwargs["dielectric"])
        self.set_concentration(float(_get(kwargs, "density")) / 10, unit=1)
