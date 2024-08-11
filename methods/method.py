import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from compounds.compound import Compound


class Method:
    def __init__(
        self,
        name,
        run_time,
        mobile_phases,
        mobile_phase_gradient_steps,
        detection,
        **kwargs
    ) -> None:
        self.name: str = name
        self.run_time: float = run_time
        self.detection_dict: dict = detection
        self.mobile_phase_dict: dict = mobile_phases
        self.mobile_phase_gradient_step_dict: dict = mobile_phase_gradient_steps
        self.__dict__ = {**self.__dict__, **kwargs}

    def create_mobile_phase_dictionary(self):
        pass

    def create_gradient_profile(self):
        pass
