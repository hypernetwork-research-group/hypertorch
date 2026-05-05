import fastjsonschema
import json
import requests

from huggingface_hub import HfApi
from importlib import resources

HIF_SCHEMA_COMMIT_SHA = "b691a3d2ec32100c0229ebe1151e9afad015c356"


def validate_hif_json(filename: str) -> bool:
    """
    Validate a JSON file against the HIF (Hypergraph Interchange Format) schema.

    Args:
        filename: Path to the JSON file to validate.

    Returns:
        ``True`` if the file is valid HIF, ``False`` otherwise.
    """
    url = f"https://raw.githubusercontent.com/HIF-org/HIF-standard/{HIF_SCHEMA_COMMIT_SHA}/schemas/hif_schema.json"
    try:
        schema = requests.get(url, timeout=10).json()
    except (requests.RequestException, requests.Timeout):
        with resources.files("hyperbench.utils.schema").joinpath("hif_schema.json").open("r") as f:
            schema = json.load(f)
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(filename, "r"))
    try:
        validator(hiftext)
        return True
    except Exception:
        return False


def get_datasets_shas(names: list[str], namespace: str = "HypernetworkRG") -> dict[str, str | None]:
    api = HfApi()
    shas: dict[str, str | None] = {}

    for dataset_name in names:
        repo_id = f"{namespace}/{dataset_name}"
        try:
            info = api.dataset_info(repo_id=repo_id)
            shas[dataset_name] = info.sha
        except Exception as e:
            shas[dataset_name] = None
            print(f"{dataset_name}: failed to retrieve SHA ({e})")

    return shas


def get_dataset_sha(dataset_name: str, namespace: str = "HypernetworkRG") -> str | None:
    api = HfApi()
    repo_id = f"{namespace}/{dataset_name}"
    try:
        info = api.dataset_info(repo_id=repo_id)
        return info.sha
    except Exception as e:
        print(f"{dataset_name}: failed to retrieve SHA ({e})")
        return None
