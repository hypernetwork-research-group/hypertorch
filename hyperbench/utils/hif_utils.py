import fastjsonschema
import json
import requests
import warnings

from huggingface_hub import HfApi
from importlib import resources
from typing import Dict, Optional

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


def get_hf_datasets_shas(
    dataset_names: list[str], namespace: str = "HypernetworkRG"
) -> Dict[str, Optional[str]]:
    shas: Dict[str, Optional[str]] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_hf_dataset_sha(dataset_name, namespace)
    return shas


def get_hf_dataset_sha(dataset_name: str, namespace: str = "HypernetworkRG") -> Optional[str]:
    api = HfApi()
    repo_id = f"{namespace}/{dataset_name}"
    try:
        info = api.dataset_info(repo_id=repo_id)
        return info.sha
    except Exception as e:
        warnings.warn(
            f"{dataset_name}: failed to retrieve SHA ({e})",
            category=UserWarning,
            stacklevel=2,
        )
        return None


def get_gh_datasets_shas(
    dataset_names: list[str],
    owner: str = "hypernetwork-research-group",
    repository: str = "datasets",
) -> Dict[str, Optional[str]]:
    shas: Dict[str, Optional[str]] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_gh_dataset_sha(dataset_name, owner, repository)
    return shas


def get_gh_dataset_sha(dataset_name: str, owner: str, repository: str) -> Optional[str]:

    OWNER = owner
    REPO = repository
    FILE_PATH = f"{dataset_name}.json.zst"

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"

    params = {
        "path": FILE_PATH,
        "per_page": 1,  # latest commit only
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        warnings.warn(
            f"{dataset_name}: failed to retrieve SHA ({e})",
            category=UserWarning,
            stacklevel=2,
        )
        return None

    data = response.json()

    if data:
        commit_sha = data[0]["sha"]
    else:
        warnings.warn(
            f"{dataset_name}: no commits found for {FILE_PATH}",
            category=UserWarning,
            stacklevel=2,
        )
        return None
    return commit_sha
