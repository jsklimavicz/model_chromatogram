from pydash import get as _get
from compounds import Compound


class Solvent(Compound):
    def __init__(self, **kwargs) -> None:
        id = [a.strip() for a in kwargs["id_list"].split(",")]
        super().__init__(**kwargs, id=id, logp=0, default_CV=0, asymmetry_addition=0)
        self.hb_acidity = float(kwargs["hb_acidity"])
        self.hb_basicity = float(kwargs["hb_basicity"])
        self.dipolarity = float(kwargs["dipolarity"])
        self.polarity = float(kwargs["polarity"])
        self.dielectric = float(kwargs["dielectric"])
        self.set_concentration(float(_get(kwargs, "density")))
