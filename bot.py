from typing import List

from model import ModelAPI
from prompt import Shot, Profile, Dialogue

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

    def refresh_prompt(self) -> None:
        raise NotImplementedError

    def generate(self) -> str:
        """
            Generate completion according to the current prompt.
        """
        raise NotImplementedError

    def add_to_history(self, utterance: str) -> None:
        """
            Append an utterance to the dialogue history.
        """
        raise NotImplementedError

    def log(self) -> None:
        raise NotImplementedError

class PatientBot(Bot):

    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: Profile,
        dialogue_history: Dialogue,
        model: ModelAPI
    ):
        super().__init__(instruction, shots, profile, dialogue_history, model)

    def inform_initial_evidence(self) -> str:
        raise NotImplementedError

    def answer(self, question: str) -> str:
        raise NotImplementedError

class DoctorBot(Bot):
    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: Profile,
        dialogue_history: Dialogue,
        model: ModelAPI
    ):
        super().__init__(instruction, shots, profile, dialogue_history, model)

    def inform_diagnosis(self) -> str:
        raise NotImplementedError
    
    def question(self, prev_answer: str) -> str:
        raise NotImplementedError