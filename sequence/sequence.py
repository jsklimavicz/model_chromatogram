import datetime
import uuid


class Sequence:
    def __init__(
        self,
        name: str,
        datavault: str,
        start_time: datetime.datetime,
        url: str,
        **kwargs,
    ) -> None:
        self.name = name
        self.start_time = start_time
        self.last_update_time = start_time
        self.url = url
        self.datavault = datavault
        self.kwargs = kwargs
        self.total_injections = 0
        self.injection_list = []

    def todict(self):
        return {
            "name": self.name,
            "datavault": self.datavault,
            "url": self.url,
            "start_time": self.start_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "total_injections": self.total_injections,
        }

    def add_injection(
        self,
        sample_name: str,
        injection_time: datetime.datetime,
        injection_uuid: uuid.uuid4,
    ) -> int:
        self.total_injections += 1
        self.last_update_time = injection_time
        self.injection_list.append(
            {
                "injection_time": injection_time,
                "injection_uuid": injection_uuid,
                "sample_name": sample_name,
                "injection_number": self.total_injections,
                "url": f"{self.url}/{injection_uuid}.json",
            }
        )
        return self.total_injections

    def get_injection_list(self):
        return self.injection_list

    def run_blank(
        self,
        injection_time: datetime.datetime,
        injection_uuid: uuid.uuid4 = uuid.uuid4(),
    ):
        self.add_injection(
            sample_name="blank",
            injection_time=injection_time,
            injection_uuid=injection_uuid,
        )

    def __iter__(self):
        return iter(self.injection_list)

    def __getitem__(self, index):
        return self.injection_list[index]

    def lookup_injection(self, uuid: str) -> dict:
        for item in self:
            if item["injection_uuid"] == uuid:
                curr_dict = self.todict()
                curr_dict["injection_number"] = item["injection_number"]
                curr_dict["injection_url"] = item["url"]
                return curr_dict
