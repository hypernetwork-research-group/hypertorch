import json
import os
import requests
import tempfile
import torch
import warnings

from huggingface_hub import hf_hub_download
from typing import Any
from torch import Tensor
from hyperbench.types import HData, HIFHypergraph
from hyperbench.utils import (
    validate_hif_json,
    decompress_zst,
    compress_to_zst,
    validate_http_url,
    write_to_disk,
    named_temporary_file,
)


GITHUB_COMMIT_SHA = "3879b2ce84750e54f984ca06ce3246dff22c71c7"


class HIFProcessor:
    """A utility class to process HIF hypergraph data into :class:`HData` format."""

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
            attr_keys: Optional list of attribute keys to encode. If provided, ensures consistent ordering and fill missing with ``0.0``.

        Returns:
            Tensor of numeric attribute values
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
    def process_hypergraph(cls, hypergraph: HIFHypergraph) -> HData:
        """
        Process the loaded hypergraph into :class:`HData` format, mapping HIF structure to tensors.

        Returns:
            The processed hypergraph data.
        """

        num_nodes = len(hypergraph.nodes)
        x = cls.__process_x(hypergraph, num_nodes)

        # Remap node IDs to 0-based contiguous IDs (using indices) matching the x tensor order
        node_id_to_idx = {node.get("node"): idx for idx, node in enumerate(hypergraph.nodes)}
        # Initialize edge_set only with edges that have incidences, so that
        # we avoid inflating edge count due to isolated nodes/missing incidences
        hyperedge_id_to_idx: dict[Any, int] = {}

        node_ids = []
        hyperedge_ids = []
        nodes_with_incidences = set()
        for incidence in hypergraph.incidences:
            node_id = incidence.get("node", 0)
            hyperedge_id = incidence.get("edge", 0)

            if hyperedge_id not in hyperedge_id_to_idx:
                # Hyperedges start from 0 and are assigned IDs in the order they are first encountered in incidences
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
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
        )

    @classmethod
    def __collect_attr_keys(cls, attr_keys: list[dict[str, Any]]) -> list[str]:
        """
        Collect unique numeric attribute keys from a list of attribute dictionaries.

        Args:
            attr_keys: List of attribute dictionaries.

        Returns:
            List of unique numeric attribute keys.
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
        # hyperedge-attr: shape [num_hyperedges, num_hyperedge_attributes]
        hyperedge_attr = None
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

        return torch.tensor(weights, dtype=torch.float)


class HIFLoader:
    """A utility class to load hypergraphs from HIF format."""

    @classmethod
    def load_from_url(cls, url: str, save_on_disk: bool = False) -> HData:
        """
        Load a hypergraph from a given URL pointing to a .json or .json.zst file in HIF format.
        Args:
            url (str): The URL to the .json or .json.zst file containing the HIF hypergraph data.
            save_on_disk (bool): Whether to save the downloaded file on disk.
        Returns:
            HData: The loaded hypergraph object.
        """
        url = validate_http_url(url)

        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download dataset from URL '{url}' with status code {response.status_code}"
            )

        if url.endswith(".json.zst"):
            zst_filename = named_temporary_file(content=response.content, suffix=".json.zst")
        elif url.endswith(".json"):
            zst_filename = named_temporary_file(content=response.content, suffix=".json")
        else:
            raise ValueError(
                f"Unsupported file format for URL '{url}'. Expected .json or .json.zst"
            )

        if zst_filename.endswith(".zst"):
            if save_on_disk:
                write_to_disk(os.path.basename(url), response.content)
            output = decompress_zst(zst_filename)
        else:  # json
            if save_on_disk:
                compressed = compress_to_zst(zst_filename)
                write_to_disk(os.path.basename(url), compressed)
            output = zst_filename

        hypergraph = cls.__extract_hif(output)
        hdata = HIFProcessor.process_hypergraph(hypergraph)
        return hdata

    @classmethod
    def load_from_path(cls, filepath: str) -> HData:
        """
        Load a hypergraph from a local file path pointing to a .json or .json.zst file in HIF format.
        Args:
            filepath (str): The local file path to the .json or .json.zst file
                containing the HIF hypergraph data.
        Returns:
            HData: The loaded hypergraph object.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File '{filepath}' does not exist.")

        if filepath.endswith(".zst"):
            output = decompress_zst(filepath)
        elif filepath.endswith(".json"):
            output = filepath
        else:
            raise ValueError(
                f"Unsupported file format for filepath '{filepath}'. Expected .json or .json.zst"
            )

        hypergraph = cls.__extract_hif(output)
        hdata = HIFProcessor.process_hypergraph(hypergraph)
        return hdata

    @classmethod
    def load_by_name(
        cls,
        dataset_name: str,
        hf_sha: str | None = None,
        save_on_disk: bool = False,
    ) -> HData:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zst_filename = os.path.join(current_dir, "datasets", f"{dataset_name}.json.zst")

        if not os.path.exists(zst_filename):
            github_url = f"https://raw.githubusercontent.com/hypernetwork-research-group/datasets/{GITHUB_COMMIT_SHA}/{dataset_name}.json.zst"
            response = requests.get(github_url, timeout=20)
            if response.status_code != 200:
                warnings.warn(
                    f"GitHub raw download failed for dataset '{dataset_name}' with status code {response.status_code}\n"
                    "Falling back to Hugging Face Hub download for dataset",
                    category=UserWarning,
                    stacklevel=2,
                )

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json.zst", delete=False
                ) as tmp_hf_file:
                    if hf_sha is not None:
                        try:
                            downloaded_path = hf_hub_download(
                                repo_id=f"HypernetworkRG/{dataset_name}",
                                filename=f"{dataset_name}.json.zst",
                                repo_type="dataset",
                                revision=hf_sha,
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Failed to download dataset '{dataset_name}' from GitHub and Hugging Face Hub. GitHub error: {response.status_code} | Hugging Face error: {e!s}"
                            ) from e
                    else:
                        raise ValueError(
                            f"Failed to download dataset '{dataset_name}' from GitHub with status code {response.status_code} and no SHA provided for Hugging Face Hub fallback."
                        )

                    with open(downloaded_path, "rb") as hf_file:
                        hf_content = hf_file.read()
                    tmp_hf_file.write(hf_content)

                response._content = hf_content

            if save_on_disk:
                os.makedirs(os.path.join(current_dir, "datasets"), exist_ok=True)
                with open(zst_filename, "wb") as f:
                    f.write(response.content)
            else:
                # Create temporary file for downloaded zst content
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json.zst", delete=False
                ) as tmp_zst_file:
                    tmp_zst_file.write(response.content)
                    zst_filename = tmp_zst_file.name

        output = decompress_zst(zst_filename)
        hypergraph = cls.__extract_hif(output)
        hdata = HIFProcessor.process_hypergraph(hypergraph)
        return hdata

    @classmethod
    def __extract_hif(cls, json_file: str) -> HIFHypergraph:
        with open(json_file) as f:
            hiftext = json.load(f)
        if not validate_hif_json(json_file):
            raise ValueError(f"Dataset from file '{json_file}' is not HIF-compliant.")
        hypergraph = HIFHypergraph.from_hif(hiftext)
        return hypergraph
