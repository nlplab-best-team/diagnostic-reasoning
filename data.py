import ast
import json
import numpy as np
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
    
    def get_evidence_set_of_initial_evidence(self, ie: str, field: str) -> set:
        """field: 'EVIDENCES' or 'EVIDENCES_ENG' or 'EVIDENCES_UNCONVERTED'"""
        return self.get_evidence_set(df=self.df[self.df.INITIAL_EVIDENCE == ie], evidence_field=field)
    
    def get_differential_of_initial_evidence(self, ie: str) -> list:
        return self.df[self.df.INITIAL_EVIDENCE == ie].PATHOLOGY.unique().tolist()
    
    def get_ddx_distribution_from_evidence(self, positives: List[str], negatives: List[str]) -> Dict[str, float]:
        ddx_count = self.get_subdf_from_evidence(positives, negatives).groupby("PATHOLOGY").size().sort_values(ascending=False)
        return (ddx_count / ddx_count.sum()).to_dict() # return percentage

    def get_subdf_from_evidence(self, positives: List[str], negatives: List[str]) -> pd.DataFrame:
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

        return self.df[self.df.EVIDENCES_ENG.map(lambda l: list_contains(l, positives) and list_excludes(l, negatives))]

    def calc_mutual_information_of_evidences(self, positives: List[str], negatives: List[str], evidence_field: str) -> Dict[str, float]:
        """Calculate mutual information of each evidence with the grounding positive/negative evidences."""
        df = self.get_subdf_from_evidence(positives, negatives)
        evds = self.get_evidence_set(df, evidence_field)
        exist_evds = set(positives) | set(negatives)
        remain_evds = evds - exist_evds
        mis = {}
        for evd in remain_evds:
            pos_df = self.get_subdf_from_evidence(positives + [evd], negatives)
            neg_df = self.get_subdf_from_evidence(positives, negatives + [evd])
            assert len(pos_df) + len(neg_df) == len(df)

            pos_prob = len(pos_df) / len(df)
            neg_prob = len(neg_df) / len(df)

            parent_entorpy = self.calc_entropy(df.groupby("PATHOLOGY").size().apply(lambda x: x / len(df)).values)
            pos_entropy = self.calc_entropy(pos_df.groupby("PATHOLOGY").size().apply(lambda x: x / len(pos_df)).values)
            neg_entropy = self.calc_entropy(neg_df.groupby("PATHOLOGY").size().apply(lambda x: x / len(neg_df)).values)

            mi = parent_entorpy - (pos_prob * pos_entropy + neg_prob * neg_entropy)
            mis[evd] = mi

        # sort by mutual information
        mis = {k: v for k, v in sorted(mis.items(), key=lambda item: item[1], reverse=True)}
        return mis

    @staticmethod
    def calc_entropy(probs: np.ndarray) -> float: # calculate Shannon entropy using natural log
        return -np.sum(probs * np.log(probs))
    
    @staticmethod
    def get_evidence_set(df: pd.DataFrame, evidence_field: str) -> set:
        evds = set()
        for evd_l in df[evidence_field].values:
            for evd in evd_l:
                evds.add(evd)
        return evds
    
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