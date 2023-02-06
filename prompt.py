from typing import List
from dataclasses import dataclass

@dataclass
class Profile(object):
    # TODO: hide text and change initialization to initialize from a patient data point
    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text

class Dialogue(object):

    def __init__(
        self,
        patient_utters: List[str] = list(),
        doctor_utters: List[str] = list(),
        doctor_first: bool = True,
        patient_prefix: str = "Patient: ",
        doctor_prefix: str = "Doctor: "
    ):
        if (doctor_first and ((len(doctor_utters) - len(patient_utters)) not in [0, 1])):
            raise ValueError("If the doctor goes first, len(doctor_utters) - len(patient_utters) should be either 1 or 0.")
        if ((not doctor_first) and ((len(patient_utters) - len(doctor_utters)) not in [0, 1])):
            raise ValueError("If the patient goes first, len(patient_utters) - len(doctor_utters) should be either 1 or 0.")

        self._patient_utters = patient_utters
        self._doctor_utters = doctor_utters
        self._doctor_first = doctor_first
        self._patient_prefix = patient_prefix
        self._doctor_prefix = doctor_prefix

    def add_patient_utter(self, utter: str) -> None:
        self._patient_utters.append(utter)
    
    def add_doctor_turn(self, utter: str) -> None:
        self._doctor_utters.append(utter)

    def __repr__(self) -> str:
        flag = self._doctor_first
        utters = list()
        for i in range(len(self._patient_utters) + len(self._doctor_utters)):
            if flag:
                utters.append(self._doctor_prefix + self._doctor_utters[i // 2])
            else:
                utters.append(self._patient_prefix + self._patient_utters[i // 2])
            flag = not flag
        return '\n'.join(utters)
        
class Shot(object):
    
    def __init__(
        self,
        profile: Profile,
        dialogue: Dialogue
    ):
        self._profile = profile
        self._dialogue = dialogue