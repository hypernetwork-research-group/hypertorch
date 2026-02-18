import fastjsonschema
import json
import requests


def validate_hif_json(filename):
    """Validate a JSON file against the HIF (Hypergraph Interchange Format) schema.

    Args:
        filename: Path to the JSON file to validate.

    Returns:
        ``True`` if the file is valid HIF, ``False`` otherwise.
    """
    url = "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json"
    try:
        schema = requests.get(url, timeout=10).json()
    except (requests.RequestException, requests.Timeout):
        with open("../schema/hif_schema.json", "r") as f:
            schema = json.load(f)
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(filename, "r"))
    try:
        validator(hiftext)
        return True
    except Exception:
        return False
