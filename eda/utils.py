# Read patient data with from csv file
import os
import ast
import json
from typing import List, Dict
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    Functions for Data Processing
"""
def read_patient_data(
    directory: Path,
    split: str
) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(directory, f'release_{split}_patients.csv'))

    return df

def read_pathology_data(
    directory: Path,
) -> Dict[str, Dict]:
    # read pathology data from json file
    with open(os.path.join(directory, 'release_conditions.json'), 'r') as f:
        data = json.load(f)

    return data

# Preprocess data
def preprocess_data(
    directory: Path,
    df: pd.DataFrame
) -> pd.DataFrame:
    # Convert string to list
    df['EVIDENCES'] = df['EVIDENCES'].apply(ast.literal_eval)
    df['DIFFERENTIAL_DIAGNOSIS'] = df['DIFFERENTIAL_DIAGNOSIS'].apply(ast.literal_eval)
    
    # Obtain only the ddx
    df['DIFFERENTIAL_DIAGNOSIS_WITHOUT_PROB'] = df['DIFFERENTIAL_DIAGNOSIS'].apply(
        lambda x: [ddx[0] for ddx in x]
    )

    # Add english pathology name
    # For pathology column
    pathology_data =  read_pathology_data(directory)
    df['PATHOLOGY_ENG'] = df['PATHOLOGY'].apply(
        lambda x: pathology_data[x]['cond-name-eng']
    )
    # For ddx column
    df['DIFFERENTIAL_DIAGNOSIS_WITHOUT_PROB_ENG'] = df['DIFFERENTIAL_DIAGNOSIS_WITHOUT_PROB'].apply(
        lambda x: [pathology_data[ddx]['cond-name-eng'] for ddx in x]
    )

    return df

"""
    Functions for Plotting
"""
def plot_catagorical_feature_distribution(
    df: pd.DataFrame,
    feature: str, # I.e., the column name
    save_path: Path = None
):
    plt.style.use('classic')
    plt.bar(
        df[feature].value_counts().index,
        df[feature].value_counts() / df[feature].value_counts().sum() * 100,
        color='lightgreen'
    )
    plt.xticks(rotation=90)
    plt.ylabel('Frequency (%)')
    plt.title(f'{feature} Distribution')
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_catagorical_feature_distribution_for_each_pathology(
    df: pd.DataFrame,
    feature: str, # I.e., the column name
    top_n: int = 50,
    save_path: Path = None
):
    gb = df.groupby('PATHOLOGY_ENG')

    plt.style.use('classic')
    fig, axs = plt.subplots(
        nrows=len(gb.groups.keys()),
        ncols=1,
        sharey=True,
        figsize=(5, 125)
    )
    fig.subplots_adjust(top=0.8, hspace=3)

    for ax, pathology in zip(axs, gb.groups.keys()):
        evidence_set = Counter()
        gb.get_group(pathology)[feature].apply(lambda x: evidence_set.update({evidence: 1 for evidence in x}))
        top_n = min(top_n, len(evidence_set))

        # plot catagorical feature distribution
        ax.bar(
            [k for k, v in evidence_set.most_common(top_n)],
            np.array([v for k, v in evidence_set.most_common(top_n)]) / sum(evidence_set.values()) * 100,
            color='lightgreen'
        )

        ax.set_title(pathology, fontsize=10)
        ax.set_ylabel('Frequency (%)', fontsize=10)
        ax.yaxis.grid(True)

        ax.tick_params('x', labelrotation=90, labelsize=8)
        ax.tick_params('y', labelsize=8)
        
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
        
"""
    For gettinng diagnosis distribution
"""
def engs_to_keys(evidences: List[str], mappings: Dict[str, str]) -> List[str]:
    return list(map(lambda eng: mappings[eng], evidences))

def get_dx_distribution(df: pd.DataFrame, positives: List[str], negatives: List[str]) -> Dict[str, float]:
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
    
    ddx_count = df[df.EVIDENCES.map(lambda l: list_contains(l, positives) and list_excludes(l, negatives))].groupby("PATHOLOGY_ENG").size().sort_values(ascending=False)
    return (ddx_count / ddx_count.sum()).to_dict() # return percentage

common_mappings = {
    "cough": "toux",
    "fever": "fievre",
    "dyspnea": "dyspn",
    "wheezing": "wheez",
    "COPD history": "j44_j42",
    "allergy tendency": "z84.89",
    "asthma": "j45",
    "smoking": "f17.210",
    "sputum": "expecto",
    "sore throat": "gorge_dlr",
    "diffuse muscle pain": "msk_dlr",
    "extremely fatigue": "fatig_ext",
    "nasal congestion or a clear runny nose": "rhino_clair",
    "GERD sensation": "pyrosis",
    "voice change": "volume_parole",
    "chest pain": "douleurxx_endroitducorps_@_c\u00f4t\u00e9_du_thorax_D_"
}
