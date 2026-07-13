import os
import shutil
import requests
import torch
import warnings

from huggingface_hub import hf_hub_download
from typing import Any
from torch import Tensor
from hypertorch.types import HData, HIFHypergraph, Task, TaskEnum
from hypertorch.utils import (
    compress_json_bytes_as_zst,
    from_bytes_to_json,
    from_file_to_json,
    from_zst_bytes_to_json,
    from_zst_file_to_json,
    validate_hif_data,
    validate_http_url,
    write_dataset_to_disk_as_zst,
    write_zst_file_to_disk,
    get_cache_dir,
)

GITHUB_COMMIT_SHA = "89ba250151bd5b1b65ba14a98dbe3dbdd72f5e25"


class HIFProcessor:
    """
    A utility class to process HIF hypergraph data into `HData` format.
    """

    @staticmethod
    def transform_attrs(
        attrs: dict[str, Any],
        attr_keys: list[str] | None = None,
    ) -> Tensor:
        """
        Extract and encode numeric attributes to tensor.

        Non-numeric attributes are discarded. Missing attributes are filled with ``0.0``.

        Args:
            attrs: Dictionary of attributes
            attr_keys: Optional list of attribute keys to encode. If provided,
                ensures consistent ordering and fill missing with ``0.0``.
                Defaults to ``None``.

        Returns:
            attrs: Tensor of numeric attribute values
        """
        numeric_attrs = {
            key: value
            for key, value in attrs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        if attr_keys is not None:
            values = [float(numeric_attrs.get(key, 0.0)) for key in attr_keys]
            return torch.tensor(values, dtype=torch.float)

        if not numeric_attrs:
            return torch.tensor([], dtype=torch.float)

        values = [float(value) for value in numeric_attrs.values()]
        return torch.tensor(values, dtype=torch.float)

    @classmethod
    def process_hypergraph(
        cls,
        hypergraph: HIFHypergraph,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
    ) -> HData:
        """
        Process the loaded hypergraph into `HData` format, mapping HIF structure to tensors.

        Returns:
            hdata: The processed hypergraph data.

        Raises:
            ValueError: If HIF node IDs are not unique or an incidence references an
                undeclared node ID.
        """
        num_nodes = len(hypergraph.nodes)
        x = cls.__process_x(hypergraph, num_nodes)
        y = cls.__process_y(hypergraph, num_nodes)
        # Remap node IDs to 0-based contiguous IDs (using indices) matching the x tensor order
        node_id_to_idx = {node.get("node"): idx for idx, node in enumerate(hypergraph.nodes)}
        if len(node_id_to_idx) != num_nodes:
            raise ValueError("HIF node IDs must be unique.")

        # Initialize edge_set only with edges that have incidences, so that
        # we avoid inflating edge count due to isolated nodes/missing incidences
        hyperedge_id_to_idx: dict[Any, int] = {}

        node_ids = []
        hyperedge_ids = []
        nodes_with_incidences = set()
        for incidence in hypergraph.incidences:
            node_id = incidence.get("node", 0)
            hyperedge_id = incidence.get("edge", 0)
            if node_id not in node_id_to_idx:
                raise ValueError(
                    f"Incidence references unknown node id {node_id!r}; "
                    "all incidence nodes must be declared in the HIF nodes list."
                )

            if hyperedge_id not in hyperedge_id_to_idx:
                # Hyperedges start from 0 and are assigned IDs in the order they are
                # first encountered in incidences
                hyperedge_id_to_idx[hyperedge_id] = len(hyperedge_id_to_idx)

            node_ids.append(node_id_to_idx[node_id])
            hyperedge_ids.append(hyperedge_id_to_idx[hyperedge_id])
            nodes_with_incidences.add(node_id_to_idx[node_id])

        # Handle isolated nodes by assigning them to a new unique hyperedge (self-loop)
        for node_idx in range(num_nodes):
            if node_idx not in nodes_with_incidences:
                new_hyperedge_id = len(hyperedge_id_to_idx)
                # Unique dummy key to reserve the index in hyperedge_set
                hyperedge_id_to_idx[f"__self_loop_{node_idx}__"] = new_hyperedge_id
                node_ids.append(node_idx)
                hyperedge_ids.append(new_hyperedge_id)

        num_hyperedges = len(hyperedge_id_to_idx)
        hyperedge_attr = cls.__process_hyperedge_attr(
            hypergraph=hypergraph,
            hyperedge_id_to_idx=hyperedge_id_to_idx,
            num_hyperedges=num_hyperedges,
        )

        hyperedge_weights = cls.__process_hyperedge_weights(
            hypergraph=hypergraph,
            hyperedge_id_to_idx=hyperedge_id_to_idx,
            num_hyperedges=num_hyperedges,
        )

        hyperedge_index = torch.tensor([node_ids, hyperedge_ids], dtype=torch.long)

        return HData(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=hyperedge_attr,
            y=y,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            task=task,
        )

    @classmethod
    def __process_y(cls, hypergraph: HIFHypergraph, num_nodes: int) -> Tensor | None:
        """
        Build the node label tensor from HIF node attributes.
        """
        node_label_keys = cls.__collect_attr_keys(
            [node.get("attrs", {}).get("label", {}) for node in hypergraph.nodes]
        )
        print(f"node_label_keys: {node_label_keys}")

        if node_label_keys:
            y = torch.stack(
                [
                    cls.transform_attrs(
                        node.get("attrs", {}).get("label", {}), attr_keys=node_label_keys
                    )
                    for node in hypergraph.nodes
                ]
            )
            return y  # shape [num_nodes, num_node_features]

        return None  # No node labels present

    @classmethod
    def __collect_attr_keys(cls, attr_keys: list[dict[str, Any]]) -> list[str]:
        """
        Collect unique numeric attribute keys from a list of attribute dictionaries.

        Args:
            attr_keys: List of attribute dictionaries.

        Returns:
            attr_keys: List of unique numeric attribute keys.
        """
        unique_keys = []
        for attrs in attr_keys:
            for key, value in attrs.items():
                if key not in unique_keys and isinstance(value, (int, float)):
                    unique_keys.append(key)

        return unique_keys

    @classmethod
    def __process_hyperedge_attr(
        cls,
        hypergraph: HIFHypergraph,
        hyperedge_id_to_idx: dict[Any, int],
        num_hyperedges: int,
    ) -> Tensor | None:
        """
        Build the hyperedge attribute matrix from HIF hyperedge attributes.

        Args:
            hypergraph: HIF hypergraph to process.
            hyperedge_id_to_idx: Mapping from HIF hyperedge IDs to contiguous indices.
            num_hyperedges: Number of hyperedges in the processed data.

        Returns:
            hyperedge_attr: Hyperedge attribute tensor, or ``None`` when no attributes exist.
        """
        hyperedge_attr = None  # shape [num_hyperedges, num_hyperedge_attributes]
        has_hyperedges = hypergraph.hyperedges is not None and len(hypergraph.hyperedges) > 0
        has_any_hyperedge_attrs = has_hyperedges and any(
            "attrs" in edge for edge in hypergraph.hyperedges
        )

        if has_any_hyperedge_attrs:
            hyperedge_id_to_attrs: dict[Any, dict[str, Any]] = {
                e.get("edge"): e.get("attrs", {}) for e in hypergraph.hyperedges
            }

            hyperedge_attr_keys = cls.__collect_attr_keys(list(hyperedge_id_to_attrs.values()))

            # Build attributes in exact order of hyperedge_set indices (0 to num_hyperedges - 1)
            hyperedge_idx_to_id = {idx: id for id, idx in hyperedge_id_to_idx.items()}

            attrs = []
            for hyperedge_idx in range(num_hyperedges):
                hyperedge_id = hyperedge_idx_to_id[hyperedge_idx]

                transformed_attrs = cls.transform_attrs(
                    # If it's a real hyperedge, get its attrs; if self-loop, get empty dict
                    attrs=hyperedge_id_to_attrs.get(hyperedge_id, {}),
                    attr_keys=hyperedge_attr_keys,
                )
                attrs.append(transformed_attrs)

            hyperedge_attr = torch.stack(attrs)

        return hyperedge_attr

    @classmethod
    def __process_x(cls, hypergraph: HIFHypergraph, num_nodes: int) -> Tensor:
        """
        Build the node feature matrix from HIF node attributes.

        Args:
            hypergraph: HIF hypergraph to process.
            num_nodes: Number of nodes in the processed data.

        Returns:
            x: Node feature matrix.
        """
        # Collect all attribute keys to have tensors of same size
        node_attr_keys = cls.__collect_attr_keys(
            [node.get("attrs", {}) for node in hypergraph.nodes]
        )

        if node_attr_keys:
            x = torch.stack(
                [
                    cls.transform_attrs(node.get("attrs", {}), attr_keys=node_attr_keys)
                    for node in hypergraph.nodes
                ]
            )
        else:
            # Fallback to ones if no node features, 1 is better as it can help during
            # training (e.g., avoid zero multiplication), especially in first epochs
            x = torch.ones((num_nodes, 1), dtype=torch.float)

        return x  # shape [num_nodes, num_node_features]

    @classmethod
    def __process_hyperedge_weights(
        cls,
        hypergraph: HIFHypergraph,
        hyperedge_id_to_idx: dict[Any, int],
        num_hyperedges: int,
    ) -> Tensor | None:
        """
        Build hyperedge weights from HIF hyperedge attributes.

        Args:
            hypergraph: HIF hypergraph to process.
            hyperedge_id_to_idx: Mapping from HIF hyperedge IDs to contiguous indices.
            num_hyperedges: Number of hyperedges in the processed data.

        Returns:
            hyperedge_weights: Hyperedge weight tensor, or ``None`` when no edge attributes exist.
        """
        has_hyperedges = hypergraph.hyperedges is not None and len(hypergraph.hyperedges) > 0
        has_any_hyperedge_attrs = has_hyperedges and any(
            "attrs" in edge for edge in hypergraph.hyperedges
        )

        # Keep old behavior for fixtures where edges have no attrs at all.
        if not has_any_hyperedge_attrs:
            return None

        # Map real edge id -> attrs (self-loops are absent and will default to 1.0)
        hyperedge_id_to_attrs: dict[Any, dict[str, Any]] = {
            e.get("edge"): e.get("attrs", {}) for e in hypergraph.hyperedges
        }

        # Build in exact hyperedge index order, defaulting missing weights to 1.0.
        hyperedge_idx_to_id = {idx: edge_id for edge_id, idx in hyperedge_id_to_idx.items()}
        weights = []
        for hyperedge_idx in range(num_hyperedges):
            edge_id = hyperedge_idx_to_id[hyperedge_idx]
            edge_attrs = hyperedge_id_to_attrs.get(edge_id, {})
            weights.append(float(edge_attrs.get("weight", 1.0)))

        return torch.tensor(weights, dtype=torch.float)  # shape [num_hyperedges,]


class HIFLoader:
    """
    A utility class to load hypergraphs from HIF format.
    """

    @classmethod
    def load_from_url(
        cls,
        url: str,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
        save_on_disk: bool = False,
    ) -> HData:
        """
        Load a hypergraph from a given URL pointing to a .json or .json.zst file in HIF format.

        Args:
            url: The URL to the .json or .json.zst file containing the HIF hypergraph data.
            task: The learning task for the loaded hypergraph.
            save_on_disk (bool): Whether to save the downloaded file on disk.

        Returns:
            hdata: The loaded hypergraph object.

        Raises:
            ValueError: If the URL cannot be downloaded, has an unsupported file format,
                or has an unexpected filename format.
        """
        url = validate_http_url(url)

        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download dataset from URL {url!r} "
                f"with status code {response.status_code}"
            )

        if not url.endswith((".json.zst", ".json")):
            raise ValueError(
                f"Unsupported file format for URL {url!r}. Expected .json or .json.zst"
            )

        if os.path.basename(url).count(".") > 2:
            raise ValueError(
                f"URL {url!r} has an unexpected filename format. "
                "Expected at most one dot in the base filename before the "
                "extension (e.g., dataset.json or dataset.json.zst)."
            )

        if url.endswith(".json.zst"):
            hif_data = from_zst_bytes_to_json(response.content)
            hdata = cls.__process_hif_data(hif_data=hif_data, task=task)
            if save_on_disk:
                write_dataset_to_disk_as_zst(
                    dataset_name=os.path.basename(url), content=response.content
                )
        else:  # json
            hif_data = from_bytes_to_json(response.content)
            hdata = cls.__process_hif_data(hif_data=hif_data, task=task)
            if save_on_disk:
                compressed_hif_data = compress_json_bytes_as_zst(response.content)

                write_dataset_to_disk_as_zst(
                    dataset_name=os.path.basename(url), content=compressed_hif_data
                )

        return hdata

    @classmethod
    def load_from_path(cls, filepath: str, task: Task = TaskEnum.HYPERLINK_PREDICTION) -> HData:
        """
        Load a hypergraph from a local file path pointing to a .json or .json.zst file in HIF
        format.

        Args:
            filepath: The local file path to the .json or .json.zst file
                containing the HIF hypergraph data.
            task: The learning task for the loaded hypergraph.

        Returns:
            hdata: The loaded hypergraph object.

        Raises:
            ValueError: If ``filepath`` does not exist or has an unsupported file format.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath!r} does not exist.")

        if filepath.endswith(".zst"):
            hif_data = from_zst_file_to_json(filepath)
        elif filepath.endswith(".json"):
            hif_data = from_file_to_json(filepath)
        else:
            raise ValueError(
                f"Unsupported format for file {filepath!r}. Expected .json or .json.zst"
            )

        return cls.__process_hif_data(hif_data=hif_data, task=task)

    @classmethod
    def load_by_name(
        cls,
        dataset_name: str,
        hf_sha: str | None = None,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
        save_on_disk: bool = False,
    ) -> HData:
        """
        Load a supported dataset by name.

        Args:
            dataset_name: Name of the dataset to load.
            task: Task type for the dataset. Defaults to "hyperlink-prediction".
            hf_sha: Optional pinned Hugging Face revision used as a fallback source.
            save_on_disk: Whether to cache the downloaded compressed dataset file.
                Defaults to ``False``.

        Returns:
            hdata: Loaded hypergraph data.

        Raises:
            ValueError: If the dataset cannot be downloaded or parsed.
        """
        cache_dir = get_cache_dir()
        output_dir = os.path.join(cache_dir, "datasets")
        zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")
        repo_root = get_cache_dir()
        hf_cache_dir = os.path.join(repo_root, "hf_cache")
        if os.path.exists(zst_filename):
            hif_data = from_zst_file_to_json(zst_filename)
            return cls.__process_hif_data(hif_data=hif_data, dataset_name=dataset_name, task=task)

        github_url = (
            f"https://raw.githubusercontent.com/hypernetwork-research-group/datasets/"
            f"{GITHUB_COMMIT_SHA}/{dataset_name}.json.zst"
        )
        response = requests.get(github_url, timeout=20)
        if response.status_code == 200:
            dataset_bytes = response.content
            hif_data = from_zst_bytes_to_json(dataset_bytes)
            hdata = cls.__process_hif_data(hif_data=hif_data, dataset_name=dataset_name, task=task)
            if save_on_disk:
                write_zst_file_to_disk(zst_filename=zst_filename, content=dataset_bytes)
            return hdata

        warnings.warn(
            f"GitHub raw download failed for dataset {dataset_name!r} "
            f"with status code {response.status_code}\n"
            "Falling back to Hugging Face Hub download for dataset",
            category=UserWarning,
            stacklevel=2,
        )

        if hf_sha is None:
            raise ValueError(
                f"Failed to download dataset {dataset_name!r} from GitHub "
                f"with status code {response.status_code} "
                f"and no SHA provided for Hugging Face Hub fallback."
            )

        try:
            token: str | None = os.getenv("HF_DOWNLOAD_TOKEN")
            token = token.strip() if token and token.strip() else None
            downloaded_path = hf_hub_download(
                repo_id=f"HypernetworkRG/{dataset_name}",
                filename=f"{dataset_name}.json.zst",
                repo_type="dataset",
                revision=hf_sha,
                cache_dir=hf_cache_dir,
                token=token,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to download dataset {dataset_name!r} from GitHub and Hugging Face Hub. "
                f"GitHub error: {response.status_code} | Hugging Face error: {e!s}."
            ) from e

        hif_data = from_zst_file_to_json(downloaded_path)
        hdata = cls.__process_hif_data(hif_data=hif_data, dataset_name=dataset_name, task=task)
        if save_on_disk:
            try:
                os.makedirs(os.path.dirname(zst_filename), exist_ok=True)
                shutil.copyfile(downloaded_path, zst_filename)
            except Exception as e:
                raise ValueError(
                    f"Failed to save downloaded dataset {dataset_name!r} to disk at "
                    f"{zst_filename!r}: {e!s}."
                ) from e

        if os.path.isdir(hf_cache_dir):
            try:
                path_prefix = f"datasets--HypernetworkRG--{dataset_name}"
                shutil.rmtree(os.path.join(hf_cache_dir, path_prefix))
                shutil.rmtree(os.path.join(hf_cache_dir, ".locks", path_prefix))
            except Exception as e:
                warnings.warn(
                    f"Failed to clean up Hugging Face Hub cache after downloading "
                    f"dataset {dataset_name!r}: {e!s}.",
                    category=UserWarning,
                    stacklevel=2,
                )
        return hdata

    @classmethod
    def __process_hif_data(
        cls,
        hif_data: dict[str, Any],
        dataset_name: str | None = None,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
    ) -> HData:
        """
        Validate and process parsed HIF data.

        Args:
            hif_data: Parsed HIF JSON data.
            dataset_name: Optional dataset name used in validation errors.
            task: Task type for the dataset. Defaults to "hyperlink-prediction".

        Returns:
            hdata: Processed hypergraph data.

        Raises:
            ValueError: If the data is not HIF-compliant.
        """
        if not validate_hif_data(hif_data):
            raise ValueError(f"Dataset {dataset_name or ''} is not HIF-compliant.")

        hypergraph = HIFHypergraph.from_hif(hif_data)
        return HIFProcessor.process_hypergraph(hypergraph=hypergraph, task=task)
