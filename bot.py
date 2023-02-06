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

        self._instruction_prefix = "<Instruction>"
        self._profile_prefix = "\n<Background information>"
        self._dialogue_prefix = "\n<History taking>"

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
        self._prompt = '\n'.join(shots + current_dialogue)
    
    def set_model_config(self) -> None:
        # TODO: delegate ModelAPI to set config
        raise NotImplementedError

    def generate(self) -> str:
        """
            Generate completion according to the current prompt.
        """
        self._refresh_prompt()
        # TODO: call the api
        results = self._prompt
        return results
        
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