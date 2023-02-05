from dataclasses import dataclass

@dataclass
class Profile(object):
    
    def __init__(self) -> None:
        pass

class Dialogue(object):

    def __init__(self):
        self._patient_utters = list()
        self._doctor_utters = list()

    def add_patient_utter(self, utter: str) -> None:
        self._patient_utters.append(utter)
    
    def add_doctor_turn(self, utter: str) -> None:
        self._doctor_utters.append(utter)

class Shot(object):
    
    def __init__(
        self,
        profile: Profile,
        dialogue: Dialogue
    ):
        self._profile = profile
        self._dialogue = dialogue