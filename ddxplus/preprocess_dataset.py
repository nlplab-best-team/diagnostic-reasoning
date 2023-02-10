import json
from pathlib import Path

def copy_field(field: str, from_d: dict, to_d: dict) -> None:
    for key in to_d:
        try:
            value = from_d[key]
            try:
                field_value = value[field]
            except KeyError:
                print(f"There is no field named {field}.")
        except KeyError:
            print(f"There is no key named {key}.")

        to_d[key][field] = field_value

if __name__ == "__main__":
    from_path = Path("./release_evidences.json")
    to_path = Path("./our_evidences_to_qa.json")

    field = "data_type"
    from_d = json.loads(from_path.read_bytes())
    to_d = json.loads(to_path.read_bytes())
    
    copy_field(field, from_d, to_d)
    to_path.write_text(json.dumps(to_d, indent=4))