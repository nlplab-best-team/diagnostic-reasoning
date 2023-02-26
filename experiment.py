import os
import re
import sys
import ast
import time
import json
import yaml
import logging
import pandas as pd
from typing import List
from pathlib import Path
from argparse import Namespace, ArgumentParser
from colorama import Fore, Style

from bot import PatientBot, DoctorBot
from utils import load_all_diagnoses
from model import ModelAPI
from prompt import Shot, Profile, PatientProfile, DoctorProfile, Dialogue

class Experiment(object):
    """
        Setup the experiment and log important information (to the terminal and to files).
    """
    def __init__(
        self,
        name: str,
        log_path: str,
        dataset_path: str,
        sample_id_path: str,
        sample_size: int,
        initial_evidence: str,
        group: str,
        pat_config_path: str,
        doc_config_path: str,
        ask_turns: int,
        resume_path: str = None,
        inform_at_turn: int = None,
        debug: bool = True # for sanity checking the script
    ):
        logging.info(f"Conducting experiment: {name}")
        # Instance variables
        self._group = group
        self._ask_turns = ask_turns
        self._resume_path = resume_path
        self._inform_at_turn = inform_at_turn
        self._debug = debug

        # Logging path
        self._log_path = Path(log_path) / name
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Select samples
        pats = pd.read_csv(dataset_path)
        if sample_id_path:
            sample_ids = json.loads(Path(sample_id_path).read_bytes())
            self._samples = pats.iloc[sample_ids]
            assert (len(self._samples) == sample_size)
        else:
            if initial_evidence:
                pats = pats[pats.INITIAL_EVIDENCE == initial_evidence]
            self._samples = pats.sample(sample_size)
        logging.info(f"Sample size: {len(self._samples)}")
        logging.info(f"Examples:")
        logging.info(self._samples.head()[["AGE", "SEX", "PATHOLOGY", "INITIAL_EVIDENCE"]])
        logging.info(f"Diagnosis stats:")
        logging.info(self._samples.groupby("PATHOLOGY").size().sort_values(ascending=False) / len(self._samples))

        # Patient
        self._pat_config = json.loads((Path(pat_config_path) / f"{group}.json").read_bytes())
        self._pat_instruction = self._pat_config["instruction"]
        self._pat_shots = [Shot(
            profile=Profile(self._pat_config["shots"][0]["profile"]),
            dialogue=Dialogue(**self._pat_config["shots"][0]["dialogue"])
        )]
        self._pat_model = ModelAPI(Namespace(**self._pat_config["model_config"]))

        # Doctor
        self._doc_config = json.loads((Path(doc_config_path) / f"{group}.json").read_bytes())
        self._doc_instruction = self._doc_config["instruction"][group]
        self._doc_profile = DoctorProfile(possible_diagnoses=load_all_diagnoses())
        self._doc_shots = [Shot(
            profile=self._doc_profile,
            dialogue=Dialogue(**json.loads((Path(doc_config_path) / group / f"{shot}.json").read_bytes()))
        ) for shot in self._doc_config["shots"]]
        self._doc_model = ModelAPI(Namespace(**self._doc_config["model_config"]))

    def estimate_cost(self) -> float:
        return 2 * len(self._samples) * 2048 * self._ask_turns * ModelAPI.cost_per_1000tokens / 1000 * ((1/2) if self._group == "baseline" else 1)
    
    def save_dialogues(self, role: str, idx: int, dialogue: Dialogue) -> None:
        assert role in ["patient", "doctor"]
        (self._log_path / f"{role}-{idx}.json").write_text(json.dumps({"doctor_utters": dialogue._doctor_utters, "patient_utters": dialogue._patient_utters}, indent=4))
    
    def pathology_to_eng(self, pathology: str):
        return PatientProfile.release_conditions[pathology]["cond-name-eng"]

    def calc_acc(self, dx_regex: str = r"the most likely diagnosis is (.*)\.", count: int = int(1e8), verbose: bool = False) -> float:
        # Load ground truths
        labels = [self.pathology_to_eng(pat.PATHOLOGY) for pat in self._samples.itertuples()][:count]
        count = min(count, len(labels))
        # Load logged results
        filenames = list()
        exist_files = set(os.listdir(self._log_path))
        for i in range(1, count + 1):
            filename = f"doctor-{i}.json"
            if filename in exist_files:
                filenames.append(filename)
            else:
                raise ValueError(f"File {filename} not found.")
        assert len(labels) == len(filenames)
        # Extract predictions
        preds = list()
        for filename, label in zip(filenames, labels):
            d = json.loads((self._log_path / filename).read_bytes())
            dx = ''
            # logging.info(filename)
            for doctor_utter in d["doctor_utters"]:
                dxs = re.findall(pattern=dx_regex, string=doctor_utter)
                if len(dxs) == 1:
                    if not dx:
                        dx = dxs[0]
                    else:
                        if (dx != dxs[0]):
                            # dx = dxs[0]
                            logging.warning(f"!!! Diagnosis {dx} not equal to {dxs[0]} in file {filename} !!!")
                elif len(dxs) > 1:
                    raise ValueError("There shouldn't be more than 1 match of 'dx_regex'. Please check.")
            if not dx:
                logging.warning(Fore.YELLOW + f"There should be at least 1 diagnosis in each dialouge. (Filename = {filename})" + Style.RESET_ALL)
            preds.append(dx)
        assert len(preds) == len(labels)
        # Comparison
        ncorrects = 0
        for i, (pred, label) in enumerate(zip(preds, labels), start=1):
            if verbose:
                    logging.info(f"{str(i).zfill(3)} -> Ground Truth: {Fore.RED + label + Style.RESET_ALL} / Prediction: {Fore.BLUE + pred + Style.RESET_ALL}{f' {Fore.GREEN}✔{Style.RESET_ALL}' if (pred == label) else ''}")
            if pred == label:
                ncorrects += 1
        logging.info(f"Correct: {ncorrects} / Predicted: {len(preds)}")
        return ncorrects / len(preds)
    
    def initialize_patient(self, pat: pd.Series) -> PatientBot:
        return PatientBot(
            instruction=self._pat_instruction,
            shots=self._pat_shots,
            profile=PatientProfile(
                sex=pat.SEX,
                age=pat.AGE,
                initial_evidence=pat.INITIAL_EVIDENCE,
                evidences=ast.literal_eval(pat.EVIDENCES)
            ),
            dialogue_history=Dialogue([], []),
            model=self._pat_model
        )
    
    def initialize_doctor(self) -> DoctorBot:
        return DoctorBot(
            instruction=self._doc_instruction,
            shots=self._doc_shots,
            profile=self._doc_profile,
            dialogue_history=Dialogue([], []),
            model=self._doc_model,
            mode=self._group
        )

    def run(self, api_interval: int, start_end: List[int] = [1, int(1e8)]) -> None:
        """
            Run the experiment. The interval between API calls is set to [api_interval] seconds.
        """
        if len(start_end) != 2:
            raise ValueError("The length of 'start_end' must be 2.")
        # Cost estimation
        cost = self.estimate_cost()
        flag = input(f"Estimated cost: {cost} USD (~{cost * 30} TWD). Continue? (Y)")
        if flag != 'Y':
            logging.info("Experiment aborted.")
            return
        
        for i, pat in enumerate(self._samples.itertuples()):
            start, end = start_end
            if (i < start - 1) or (i >= end): # Start with the [start]-th example and end with the [end]-th example
                continue
            logging.info(f"===== Running with sample {i + 1} -> PATHOLOGY: {self.pathology_to_eng(pat.PATHOLOGY)} =====")
            # Initialize the patient
            patient = self.initialize_patient(pat)
            logging.info(f"Patient initialized. Chief complaint: {patient.inform_initial_evidence()}; Basic info: {patient.inform_basic_info()}")
            # Initialize the doctor
            doctor = self.initialize_doctor()
            logging.info(f"Doctor initialized.")

            if self._resume_path is None:
                # History taking
                a = patient.answer(question=doctor.greeting(), answer=patient.inform_initial_evidence())
                r, q = doctor.ask(prev_answer=a, question=doctor.ask_basic_info())
                a = patient.answer(question=q, answer=patient.inform_basic_info())
                
                for j in range(self._ask_turns):
                    # Doctor: ask a question
                    r, q = doctor.ask(prev_answer=a, question=f"{'R [Question] ' if (self._group == 'reasoning') else ''}Q{j}" if self._debug else '')
                    time.sleep(api_interval)
                    # Patient: give an answer
                    a = patient.answer(question=q, answer=f"A{j}" if self._debug else '')
                    time.sleep(api_interval)
                    # Logging
                    logging.info(f"Turn {j + 1} completed:")
                    logging.info(f"Doctor: [reasoning] {r}[question] {q}")
                    logging.info(f"Patient: {a}")
                    # Save dialogues
                    self.save_dialogues(role="patient", idx=i + 1, dialogue=patient._dialogue_history)
                    self.save_dialogues(role="doctor", idx=i + 1, dialogue=doctor._dialogue_history)
            else: # If there is a resume_path -> load previous dialogues and jump to inform_diagnosis
                doc_dials = json.loads((Path(self._resume_path) / f"doctor-{i + 1}.json").read_bytes())
                dial_length = self._inform_at_turn + Dialogue.greeting_length
                doctor.set_dialogue_history(
                    Dialogue(
                        patient_utters=doc_dials["patient_utters"][:dial_length - 1], # preserve for the parameter 'prev_answer' in doctor.inform_diagnosis()
                        doctor_utters=doc_dials["doctor_utters"][:dial_length]
                    )
                )
                a = doc_dials["patient_utters"][dial_length - 1]

            inform = doctor.inform_diagnosis(prev_answer=a, utter='D' if self._debug else '')
            time.sleep(api_interval)
            logging.info(f"Doctor: {inform}")
            logging.info(f"===== Sample {i + 1} completed =====")
            # Save dialogues
            self.save_dialogues(role="patient", idx=i + 1, dialogue=patient._dialogue_history)
            self.save_dialogues(role="doctor", idx=i + 1, dialogue=doctor._dialogue_history)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--config_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1
    )
    parser.add_argument(
        "--end",
        type=int,
        default=int(1e8)
    )
    parser.add_argument(
        "--eval",
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    args = parse_args()
    
    with open(args.config_file, mode="rt") as f:
        config = yaml.safe_load(f)
        logging.info(f"Configuration loaded: {json.dumps(config, indent=4)}")
    
    exp = Experiment(**config, debug=args.debug)
    if not args.eval:
        exp.run(api_interval=args.interval, start_end=[args.start, args.end])
    else:
        acc = exp.calc_acc(count=config["sample_size"], verbose=True)
        print(f"Accuracy: {acc * 100:.2f}%")