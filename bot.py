import re
import openai
from typing import List
from pathlib import Path
from colorama import Fore, Style

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
        
    def get_prompt(self) -> str:
        self._refresh_prompt()
        return self._prompt

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
    
    def set_dialogue_history(self, dialogue: Dialogue) -> None:
        self._dialogue_history = dialogue

    def generate(self, prefix: str = '') -> str:
        """
            Generate completion according to the current prompt and an additional prefix.
        """
        self._refresh_prompt()
        try:
            res = self._model.complete(prompt=self._prompt + prefix)["choices"][0]["text"]
        except openai.error.InvalidRequestError:
            print(Fore.YELLOW + "Invalid request. Stop the dialogue." + Style.RESET_ALL)
            res = "(End of dialogue.)"
        return res

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
        return self._profile.initial_evidence
    
    def inform_basic_info(self) -> str:
        return self._profile.basic_info

    def answer(self, question: str, answer: str = '') -> str:
        self._dialogue_history.add_doctor_utter(question)
        if not answer:
            answer = self.generate(prefix="\nPatient: ").strip()
        self._dialogue_history.add_patient_utter(answer)
        return answer

class DoctorBot(Bot):
    ask_prefix = "[ask]"
    inform_prefix = "[inform]"
    question_prefix = "[Question]"

    def __init__(
        self,
        instruction: str,
        shots: List[Shot],
        profile: DoctorProfile,
        dialogue_history: Dialogue,
        model: ModelAPI,
        mode: str = "" # Either "baseline" or "reasoning"
    ):
        if mode not in ["baseline", "reasoning"]:
            raise ValueError("The argument 'mode' should be 'baseline' or 'reasoning'.")
        
        super().__init__(instruction, shots, profile, dialogue_history, model)

        self._mode = mode

        self._instruction_prefix = "\n<Instruction>"
        self._profile_prefix = "\n<Background knowledge>"
        self._dialogue_prefix = "\n<History taking>"

        self._greeting = "How may I help you today?"
        self._basic_info_q = "What's your sex and age?"

    def _refresh_prompt(self) -> None:
        prefix = '\n'.join([self._instruction_prefix, self._instruction, self._profile_prefix, str(self._profile)])
        shots = [
            (
                self._dialogue_prefix + '\n' +
                str(shot._dialogue)
            ) for shot in self._shots
        ]
        current_dialogue = [
            self._dialogue_prefix + '\n' +
            str(self._dialogue_history)
        ]
        self._prompt = prefix + '\n\n' + '\n'.join(shots + current_dialogue).strip()

    def greeting(self) -> str:
        self._dialogue_history.add_doctor_utter(self.ask_prefix + ' ' + self._greeting)
        return self._greeting
    
    def ask_basic_info(self) -> str:
        return self._basic_info_q
    
    def ask(self, prev_answer: str, question: str = '') -> str:
        self._dialogue_history.add_patient_utter(prev_answer)
        if not question:
            question = self.generate(prefix=f"\nDoctor: {self.ask_prefix} ").strip()
        self._dialogue_history.add_doctor_utter(f"{self.ask_prefix} {question}")
                 
        if (self._mode == "baseline") or (question == self.ask_basic_info()):
            reasoning = ''
        elif self._mode == "reasoning":
            question_index = question.find(self.question_prefix)
            if question_index == -1:
                print(f"Question index not found by prefix {self.question_prefix} in question {question}.")
                return question, ''
            reasoning = question[0:question_index]
            question = question[question_index + len(self.question_prefix) + 1:]
        else:
            raise ValueError(f"self._model should be either 'baseline' or 'reasoning'.")
            
        return reasoning, question
    
    def inform_diagnosis(self, prev_answer: str, utter: str = '') -> str:
        self._dialogue_history.add_patient_utter(prev_answer)
        if not utter:
            utter = self.generate(prefix=f"\nDoctor: {self.inform_prefix} ").strip()
        self._dialogue_history.add_doctor_utter(f"{self.inform_prefix} {utter}")
        return utter