import fastjsonschema
import json
import requests
import warnings

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
        with (
            resources.files("hyperbench.utils.schema")
            .joinpath("hif_schema.json")
            .open("r") as f
        ):
            schema = json.load(f)
    validator = fastjsonschema.compile(schema)

    with open(filename) as f:
        hiftext = json.load(f)
        try:
            validator(hiftext)
            return True
        except Exception:
            return False


def get_hf_datasets_shas(
    dataset_names: list[str], namespace: str = "HypernetworkRG"
) -> dict[str, str | None]:
    shas: dict[str, str | None] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_hf_dataset_sha(dataset_name, namespace)
    return shas


def get_hf_dataset_sha(
    dataset_name: str, namespace: str = "HypernetworkRG"
) -> str | None:
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
) -> dict[str, str | None]:
    shas: dict[str, str | None] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_gh_dataset_sha(dataset_name, owner, repository)
    return shas


def get_gh_dataset_sha(dataset_name: str, owner: str, repository: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repository}/commits"
    file_path = f"{dataset_name}.json.zst"

    params = {
        "path": file_path,
        "per_page": 1,  # Latest commit only
    }

    try:
        response = requests.get(url, params=params, timeout=10)
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
            f"{dataset_name}: no commits found for {file_path}",
            category=UserWarning,
            stacklevel=2,
        )
        return None
    return commit_sha
