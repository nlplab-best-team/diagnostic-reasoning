import sys
import ast
import time
import json
import yaml
import logging
import pandas as pd
from pathlib import Path

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
        debug: bool = True # for sanity checking the script
    ):
        logging.info(f"Conducting experiment: {name}")
        self._group = group
        self._ask_turns = ask_turns
        self._debug = debug
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

        # Patient
        self._pat_config = json.loads((Path(pat_config_path) / f"{name}.json").read_bytes())
        logging.info(f"Patient configuration loaded: {json.dumps(self._pat_config, indent=4)}")
        self._pat_instruction = self._pat_config["instruction"]
        self._pat_shots = [Shot(
            profile=Profile(self._pat_config["shots"][0]["profile"]),
            dialogue=Dialogue(**self._pat_config["shots"][0]["dialogue"])
        )]
        self._pat_model = ModelAPI(self._pat_config["model_config"])

        # Doctor
        self._doc_config = json.loads((Path(doc_config_path) / f"{name}.json").read_bytes())
        self._doc_instruction = self._doc_config["instruction"][group]
        self._doc_profile = DoctorProfile(possible_diagnoses=load_all_diagnoses())
        self._doc_shots = [Shot(
            profile=self._doc_profile,
            dialogue=Dialogue(**json.loads((Path(doc_config_path) / group / f"{shot}.json").read_bytes()))
        ) for shot in self._doc_config["shots"]]
        self._doc_model = ModelAPI(self._doc_config["model_config"])

    def estimate_cost(self) -> float:
        raise NotImplementedError
    
    def run(self, api_interval: int) -> None:
        """
            Run the experiment. The interval between API calls is set to [api_interval] seconds.
        """
        for i, pat in enumerate(self._samples.itertuples()):
            logging.info(f"===== Running with sample {i + 1} -> PATHOLOGY: {pat.PATHOLOGY} =====")
            # Initialize the patient
            patient = PatientBot(
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
            logging.info(f"Patient initialized. Chief complaint: {patient.inform_initial_evidence()}; Basic info: {patient.inform_basic_info()}")
            # Initialize the doctor
            doctor = DoctorBot(
                instruction=self._doc_instruction,
                shots=self._doc_shots,
                profile=self._doc_profile,
                dialogue_history=Dialogue([], []),
                model=self._doc_model,
                mode=self._group
            )
            logging.info(f"Doctor initialized.")
            logging.info(doctor.get_prompt())
            # logging.info(patient._dialogue_history is doctor._dialogue_history)
            # History taking
            a = patient.answer(question=doctor.greeting(), answer=patient.inform_initial_evidence())
            q = doctor.ask(prev_answer=a, question=doctor.ask_basic_info())
            a = patient.answer(question=q, answer=patient.inform_basic_info())
            for j in range(self._ask_turns):
                r, q = doctor.ask(prev_answer=a, question=f"{'R [Question] ' if (self._group == 'reasoning') else ''}Q{j}" if self._debug else '')
                time.sleep(api_interval)
                a = patient.answer(question=q, answer=f"A{j}" if self._debug else '')
                time.sleep(api_interval)
                logging.info(f"Turn {j + 1} completed:")
                logging.info(f"Doctor: [reasoning] {r}[question] {q}")
                logging.info(f"Patient: {a}")
            doctor.inform_diagnosis(prev_answer=a)
            logging.info(f"===== Sample {i + 1} completed =====")
            (self._log_path / f"{i + 1}.txt").write_text(str(doctor._dialogue_history))
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    with open("./config.yml", mode="rt") as f:
        config = yaml.safe_load(f)
        logging.info(f"Configuration loaded: {json.dumps(config, indent=4)}")
    
    exp = Experiment(**config, debug=True)
    exp.run(api_interval=0)