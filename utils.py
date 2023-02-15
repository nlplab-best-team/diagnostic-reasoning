import json
from typing import List
from pathlib import Path

def calc_num_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])

def load_all_diagnoses(file: str = (Path(__file__).parent / "./ddxplus/release_conditions.json")) -> List[str]:
    d = json.loads(Path(file).read_bytes())
    dxs = set()
    for _, v in d.items():
        dxs.add(v["cond-name-eng"])
    return list(dxs)