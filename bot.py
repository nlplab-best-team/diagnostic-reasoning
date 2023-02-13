from typing import List
from pathlib import Path

from model import ModelAPI
from prompt import Shot, Profile, PatientProfile, DoctorProfile, Dialogue

class Bot(object):

    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: Profile,
        dialogue_history: Dialogue,
        model: ModelAPI
    ):
        self._prompt = ""
        self._instruction = instruction
        self._shots = shots
        self._profile = profile
        self._dialogue_history = dialogue_history
        self._model = model

        self._instruction_prefix = "\n<Instruction>"
        self._profile_prefix = "\n<Background information>"
        self._dialogue_prefix = "\n<History taking>"
        
        self._refresh_prompt()

    def _refresh_prompt(self) -> None:
        shots = [
            (
                self._instruction_prefix + '\n' +
                self._instruction + '\n' +
                self._profile_prefix + '\n' +
                str(shot._profile) + '\n' +
                self._dialogue_prefix + '\n' +
                str(shot._dialogue)
            ) for shot in self._shots
        ]
        current_dialogue = [
            self._instruction_prefix + '\n' +
            self._instruction + '\n' +
            self._profile_prefix + '\n' +
            str(self._profile) + '\n' +
            self._dialogue_prefix + '\n' +
            str(self._dialogue_history)
        ]
        self._prompt = '\n'.join(shots + current_dialogue).strip()
    
    def set_model_config(self) -> None:
        # TODO: delegate ModelAPI to set config
        raise NotImplementedError

    def generate(self, prefix: str = '') -> str:
        """
            Generate completion according to the current prompt and an additional prefix.
        """
        self._refresh_prompt()
        # TODO: call the api
        res = self._model.complete(prompt=self._prompt + prefix)
        return res["choices"][0]["text"]
        
    # def add_to_history(self, utterance: str) -> None:
    #     """
    #         Append an utterance to the dialogue history.
    #     """
    #     raise NotImplementedError

    def log(self, mode: str, file: str) -> None:
        if mode == "text":
            Path(file).write_text(self._prompt)

class PatientBot(Bot):

    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: PatientProfile,
        dialogue_history: Dialogue,
        model: ModelAPI
    ):
        super().__init__(instruction, shots, profile, dialogue_history, model)

    def inform_initial_evidence(self) -> str:
        raise NotImplementedError

    def answer(self, question: str, answer: str = '') -> str:
        self._dialogue_history.add_doctor_utter(question)
        if not answer:
            answer = self.generate(prefix="\nPatient:")
        return answer.strip()

class DoctorBot(Bot):
    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: DoctorProfile,
        dialogue_history: Dialogue,
        model: ModelAPI
    ):
        super().__init__(instruction, shots, profile, dialogue_history, model)

    def inform_diagnosis(self) -> str:
        raise NotImplementedError
    
    def question(self, prev_answer: str, question: str = '') -> str:
        self._dialogue_history.add_patient_utter(prev_answer)
        if not question:
            question = self.generate()
        return question.strip()