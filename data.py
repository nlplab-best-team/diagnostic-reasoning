import ast
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path

class DDxDataset(object):

    def __init__(
        self,
        csv_path: str,
        pathology_info_path: str,
        evidences_info_path: str
    ):
        df = pd.read_csv(csv_path)
        pathology_info = json.loads(Path(pathology_info_path).read_text())
        evidences_info = json.loads(Path(evidences_info_path).read_text())
        # Preprocess dataframe
        # convert evidences and diagnoses to English
        # 1. convert string to list ("DIFFERENTIAL_DIAGNOSIS", "EVIDENCES")
        df.DIFFERENTIAL_DIAGNOSIS = df.DIFFERENTIAL_DIAGNOSIS.apply(ast.literal_eval)
        df.EVIDENCES = df.EVIDENCES.apply(ast.literal_eval)
        # 2. convert pathology to English ("PATHOLOGY", "DIFFERENTIAL_DIAGNOSIS")
        df.PATHOLOGY = df.PATHOLOGY.apply(lambda dx: pathology_info[dx]['cond-name-eng'])
        df.DIFFERENTIAL_DIAGNOSIS = df.DIFFERENTIAL_DIAGNOSIS.apply(lambda dxs_probs: [[pathology_info[dx_prob[0]]['cond-name-eng'], dx_prob[1]] for dx_prob in dxs_probs])
        # 3. convert evidences to English ("INITIAL_EVIDENCE", "EVIDENCES")
        df.INITIAL_EVIDENCE = df.INITIAL_EVIDENCE.apply(lambda ie: evidences_info[ie]["abbr_en"].lower())
        # EVIDENCES: currently only binary evidences are supported
        # TODO: M and C are not supported yet -> add a temporary field "EVIDENCES_UNCONVERTED"
        df["EVIDENCES_ENG"] = df.EVIDENCES.apply(self.convert_evidence_to_eng(evidences_info, "eng"))
        df["EVIDENCES_UNCONVERTED"] = df.EVIDENCES.apply(self.convert_evidence_to_eng(evidences_info, "unconverted"))
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def get_evidence_set_of_initial_evidence(self, ie: str, field: str) -> list:
        """field: 'EVIDENCES' or 'EVIDENCES_ENG' or 'EVIDENCES_UNCONVERTED'"""
        evds = set()
        for evd_l in self.df[self.df.INITIAL_EVIDENCE == ie][field].values:
            for evd in evd_l:
                evds.add(evd)
        return list(evds)
    
    def get_differential_of_initial_evidence(self, ie: str) -> list:
        return self.df[self.df.INITIAL_EVIDENCE == ie].PATHOLOGY.unique().tolist()
    
    def get_ddx_distribution_from_evidence(self, positives: List[str], negatives: List[str]) -> Dict[str, float]:
        def list_contains(l: List[str], contents: List[str]) -> bool:
            s = set(l)
            for content in contents:
                if not (content in s):
                    return False
            return True

        def list_excludes(l: List[str], contents: List[str]) -> bool:
            s = set(l)
            for content in contents:
                if content in s:
                    return False
            return True

        ddx_count = self.df[self.df.EVIDENCES_ENG.map(lambda l: list_contains(l, positives) and list_excludes(l, negatives))].groupby("PATHOLOGY").size().sort_values(ascending=False)
        return (ddx_count / ddx_count.sum()).to_dict() # return percentage
    
    @staticmethod
    def convert_evidence_to_eng(evidences_info: Dict[str, Dict], mode: str):
        def convert_evidence(evds: list):
            eng, unconverted = [], []
            for evd in evds:
                if evd in evidences_info: # binary-like evidence
                    eng.append(evidences_info[evd]["abbr_en"].lower())
                else:
                    unconverted.append(evd)
            if mode == "eng":
                return eng
            elif mode == "unconverted":
                return unconverted
            else:
                raise ValueError(f"mode should be either 'eng' or 'unconverted', but got {mode}")
        return convert_evidence