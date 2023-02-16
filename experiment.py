import sys
import json
import yaml
import logging
import pandas as pd
from pathlib import Path

class Experiment(object):
    """
        Setup the experiment and log important information (to the terminal and to files).
    """
    def __init__(
        self,
        name: str,
        dataset_path: str,
        sample_id_path: str,
        sample_size: int,
        initial_evidence: str,
        group: str,
        pat_exemplar_path: str,
        doc_exemplar_path: str,
        ask_turns: int
    ):
        logging.info(f"Conducting experiment: {name}")
        
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
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    with open("./config.yml", mode="rt") as f:
        config = yaml.safe_load(f)
        logging.info(f"Configuration loaded: {json.dumps(config, indent=4)}")
    
    exp = Experiment(**config)