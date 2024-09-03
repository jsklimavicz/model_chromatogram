from pydash import get as _get


class ProcessingMethod:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.peak_identification = _get(kwargs, "peak_identification")

    def get_peak_identification(self):
        return self.peak_identification

    def todict(self):
        return self.kwargs
