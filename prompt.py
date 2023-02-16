import re
import json
from typing import List
from pathlib import Path
from dataclasses import dataclass

class Profile(object):
    # TODO: hide text and change initialization to initialize from a patient data point
    def __init__(self, text: str = ''):
        self._text = text

    def __repr__(self) -> str:
        return self._text

class PatientProfile(Profile):
    delimiter = "_@_"
    release_evidences = json.loads((Path(__file__).parent / "ddxplus/release_evidences.json").read_bytes())
    evidence2desc = json.loads((Path(__file__).parent / "ddxplus/our_evidences_to_qa.json").read_bytes())
    desc_field = "affirmative_en"
    
    def __init__(
        self,
        sex: str,
        age: int,
        initial_evidence: str,
        evidences: List[str]
    ):
        if sex not in ['M', 'F']:
            raise ValueError(f"Sex should be either 'M' or 'F'")
        self._sex = sex
        self._age = age
        self._initial_evidence = initial_evidence
        self._parsed = self._parse_evidences(evidences)
        
    def __repr__(self) -> str:
        """
            Convert parsed evidences to text representation.
        """
        sents = list()
        # Sex & age
        sex_name = 'Male' if self._sex == 'M' else 'Female'
        sent = f"Sex: {sex_name}, Age: {str(self._age)}"
        sents.append(sent)
        
        # binary
        for key in self._parsed['B']:
            sent = self.evidence2desc[key][self.desc_field]
            sents.append(f"- {sent}")
        # categorical
        for key, value in self._parsed['C'].items():
            sent = self.evidence2desc[key][self.desc_field]
            sent = re.sub(pattern=r"\[choice\]", string=sent, repl=str(value))
            sents.append(f"- {sent}")
        # multichoice
        for key, values in self._parsed['M'].items():
            header = self.evidence2desc[key][self.desc_field]
            sents.append(f"- {header}")
            for value in values:
                item = self.release_evidences[key]["value_meaning"][value]["en"]
                sents.append(f"* {item}")
        
        return '\n'.join(sents)
        
    def _parse_evidences(self, evidences: List[str]) -> dict:
        """
            Parse the evidences into the following format:
            {
                "B": [key1, key2, ...],
                "C": {
                    key1: value1,
                    key2: value2,
                    ...
                },
                "M": {
                    key1: [value1, value2, ...],
                    key2: [value1, value2, ...],
                    ...
                }
            }
        """
        d = {'B': [], 'C': {}, 'M': {}}
        for evidence in evidences:
            evidence = evidence.split(self.delimiter)
            evidence_name = evidence[0]
            if evidence_name not in self.release_evidences:
                raise KeyError(f"There is no evidence called {evidence_name}")
            data_type = self.release_evidences[evidence_name]["data_type"]
            
            if len(evidence) == 1: # binary
                if data_type != 'B':
                    raise ValueError(f"The date_type of evidence {evidence_name} should be binary (B). Please check!")
                d[data_type].append(evidence_name)
                
            elif len(evidence) == 2: # categorical or multichoice
                if data_type not in ['C', 'M']:
                    raise ValueError(f"The date_type of evidence {evidence_name} should be either categorical (C) or multichoice (M). Please check!")
                evidence_value = evidence[1]
                if data_type == 'C': # categorical
                    d[data_type][evidence_name] = evidence_value
                else: # multichoice
                    d[data_type][evidence_name] = d[data_type].get(evidence_name, []) + [evidence_value]
                    
            else:
                raise ValueError(f"After spliiting with {self.delimiter}, the length of {evidence} should be either 1 or 2.")
        
        return d
    
    @property
    def initial_evidence(self) -> str:
        return self.evidence2desc[self._initial_evidence][self.desc_field]
    
    @property
    def basic_info(self) -> str:
        return f"I am a {self._age}-year-old {'male' if self._sex == 'M' else 'female'}."

class DoctorProfile(Profile):
    diagnoses_prefix: str = "Possible diagnoses: "
    delimiter: str = ", "
    
    def __init__(self, possible_diagnoses: List[str]):
        self._possible_diagnoses = possible_diagnoses
        self._text = self._parse_diagnoses(possible_diagnoses)
    
    def _parse_diagnoses(self, diagnoses: List[str]) -> str:
        return self.diagnoses_prefix + self.delimiter.join(diagnoses)

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
    
    def add_doctor_utter(self, utter: str) -> None:
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