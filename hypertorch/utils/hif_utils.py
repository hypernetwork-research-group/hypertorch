import fastjsonschema
import json
from pathlib import Path
import requests
import warnings

from huggingface_hub import HfApi
from importlib import resources
from typing import Any

from hypertorch.utils.file_utils import get_cache_dir
from hypertorch.utils.url_utils import validate_http_url

HIF_SCHEMA_COMMIT_SHA = "b691a3d2ec32100c0229ebe1151e9afad015c356"


def validate_hif_data(hif_data: dict[str, Any]) -> bool:
    """
    Validate a Python object against the HIF schema.

    Args:
        hif_data: Parsed HIF data to validate.

    Returns:
        valid: ``True`` when the data conforms to the schema, otherwise ``False``.
    """
    schema = __load_hif_schema()
    validator = fastjsonschema.compile(schema)
    try:
        validator(hif_data)
        return True
    except Exception:
        return False


def validate_hif_json(filename: str) -> bool:
    """
    Validate a JSON file against the HIF (Hypergraph Interchange Format) schema.

    Args:
        filename: Path to the JSON file to validate.

    Returns:
        valid: ``True`` if the file is valid HIF, ``False`` otherwise.

    Raises:
        ValueError: If the JSON file cannot be read.
    """
    try:
        with open(filename, encoding="utf-8") as f:
            hiftext = json.load(f)
            return validate_hif_data(hiftext)
    except Exception as e:
        raise ValueError(f"Failed to read JSON file {filename!r}: {e!s}.") from e


def get_hf_datasets_shas(
    dataset_names: list[str], namespace: str = "HypernetworkRG"
) -> dict[str, str | None]:
    """
    Retrieve Hugging Face dataset commit SHAs for multiple datasets.

    Args:
        dataset_names: Dataset names to query.
        namespace: Hugging Face namespace containing the datasets.
            Defaults to ``"HypernetworkRG"``.

    Returns:
        shas: Mapping from dataset name to commit SHA, or ``None`` when unavailable.
    """
    shas: dict[str, str | None] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_hf_dataset_sha(dataset_name, namespace)
    return shas


def get_hf_dataset_sha(dataset_name: str, namespace: str = "HypernetworkRG") -> str | None:
    """
    Retrieve the latest Hugging Face commit SHA for a dataset.

    Args:
        dataset_name: Dataset name to query.
        namespace: Hugging Face namespace containing the dataset.
            Defaults to ``"HypernetworkRG"``.

    Returns:
        sha: Dataset repository commit SHA, or ``None`` when unavailable.
    """
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
    """
    Retrieve GitHub commit SHAs for multiple compressed dataset files.

    Args:
        dataset_names: Dataset file stems to query.
        owner: GitHub repository owner.
        repository: GitHub repository name. Defaults to ``"datasets"``.

    Returns:
        shas: Mapping from dataset name to commit SHA, or ``None`` when unavailable.
    """
    shas: dict[str, str | None] = {}

    for dataset_name in dataset_names:
        shas[dataset_name] = get_gh_dataset_sha(dataset_name, owner, repository)
    return shas


def get_gh_dataset_sha(dataset_name: str, owner: str, repository: str) -> str | None:
    """
    Retrieve the latest GitHub commit SHA for a dataset file.

    Args:
        dataset_name: Dataset file stem without the ``.json.zst`` suffix.
        owner: GitHub repository owner.
        repository: GitHub repository name.

    Returns:
        sha: Latest commit SHA for the dataset file, or ``None`` when unavailable.
    """
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


def __load_hif_schema() -> dict[str, Any]:
    """
    Load the HIF schema from cache, bundled resources, or the pinned remote URL.

    Returns:
        schema: Parsed HIF JSON schema.

    Raises:
        RuntimeError: If the schema cannot be loaded from any source.
    """
    cache_dir = get_cache_dir()
    cached_hif_schema = Path(cache_dir) / "hif_schema.json"
    try:
        if cached_hif_schema.exists():
            with cached_hif_schema.open("r", encoding="utf-8") as f:
                return json.load(f)

        with (
            resources.files("hypertorch.utils.schema")
            .joinpath("hif_schema.json")
            .open("r", encoding="utf-8") as f
        ):
            return json.load(f)
    except Exception:
        try:
            url = f"https://raw.githubusercontent.com/HIF-org/HIF-standard/{HIF_SCHEMA_COMMIT_SHA}/schemas/hif_schema.json"
            validate_http_url(url)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            schema = response.json()
            with cached_hif_schema.open("w", encoding="utf-8") as f:
                json.dump(schema, f)
            return schema
        except (requests.RequestException, requests.Timeout) as e:
            raise RuntimeError(
                "Failed to load HIF schema from both local file and remote URL. "
            ) from e
