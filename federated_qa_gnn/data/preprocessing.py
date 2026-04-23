from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("federated_qa")


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def balance_datasets(
    client_data: Dict[int, List[Dict[str, Any]]],
    seed: int = 42,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[int]], int]:
    """
    Subsample each client's data to the minimum number of entries across clients.

    Sampling is at the input-entry level (not qa_pair level). All qa_pairs
    within a kept entry are retained.

    Returns:
        balanced: dict mapping client_id → balanced list of entries
        kept_original_indices: dict mapping client_id → original positional indices kept
        n_min: the target size applied to every client
    """
    rng = random.Random(seed)
    sizes = {k: len(v) for k, v in client_data.items()}
    n_min = min(sizes.values())

    logger.info(f"Dataset balancing — n_min = {n_min}")
    for cid, n in sizes.items():
        logger.info(f"  Client {cid}: {n} entries → {n_min} kept, {n - n_min} discarded")

    balanced: Dict[int, List[Dict[str, Any]]] = {}
    kept_indices: Dict[int, List[int]] = {}
    for client_id, data in client_data.items():
        all_indices = list(range(len(data)))
        if len(data) > n_min:
            chosen = rng.sample(all_indices, n_min)
            chosen.sort()
        else:
            chosen = all_indices
        kept_indices[client_id] = chosen
        balanced[client_id] = [data[i] for i in chosen]

    return balanced, kept_indices, n_min


def split_data(
    data: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split entry indices 70 / 15 / 15 (train / val / test).

    Split is at the input-entry level to prevent context leakage.

    Returns:
        train_indices, val_indices, test_indices — all as indices into `data`
    """
    rng = random.Random(seed)
    n = len(data)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def flatten_qa_pairs(
    data: List[Dict[str, Any]],
    indices: List[int],
) -> List[Dict[str, str]]:
    """
    Flatten qa_pairs so that each (context, qa_pair) becomes one sample.

    Returns a list of dicts with keys: context, question, answer.
    """
    samples: List[Dict[str, str]] = []
    for idx in indices:
        entry = data[idx]
        context = entry.get("clean_context", "")
        for qa in entry.get("qa_pairs", []):
            samples.append(
                {
                    "context": context,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                }
            )
    return samples


def prepare_all_data(
    client_configs: List[Any],  # List[ClientConfig]
    output_dir: str = "outputs/",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full pipeline: load → balance → split → flatten → save indices.

    Returns a dict with keys:
        n_min, raw_counts, balanced_counts, train_qa_counts,
        client_data (balanced lists), splits (per-client train/val/test flat samples)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    raw_data: Dict[int, List[Dict[str, Any]]] = {}
    for cfg in client_configs:
        path = Path(cfg.data_path)
        raw_data[cfg.client_id] = load_json(path)
        logger.info(f"Loaded client {cfg.client_id}: {len(raw_data[cfg.client_id])} entries from {path}")

    raw_counts = {cid: len(d) for cid, d in raw_data.items()}

    # Balance
    balanced_data, kept_original_indices_map, n_min = balance_datasets(raw_data, seed=seed)
    balanced_counts = {cid: len(d) for cid, d in balanced_data.items()}

    # Split and flatten
    split_indices: Dict[str, Any] = {"n_min": n_min}
    client_splits: Dict[int, Dict[str, List[Dict[str, str]]]] = {}
    train_qa_counts: Dict[int, int] = {}

    for cfg in client_configs:
        cid = cfg.client_id
        data = balanced_data[cid]
        kept_original_indices = kept_original_indices_map[cid]

        train_idx, val_idx, test_idx = split_data(data, seed=seed)

        train_samples = flatten_qa_pairs(data, train_idx)
        val_samples = flatten_qa_pairs(data, val_idx)
        test_samples = flatten_qa_pairs(data, test_idx)

        train_qa_counts[cid] = len(train_samples)
        client_splits[cid] = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        split_indices[f"client_{cid}"] = {
            "kept_indices": kept_original_indices,
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        logger.info(
            f"Client {cid}: train={len(train_samples)} qa-pairs, "
            f"val={len(val_samples)}, test={len(test_samples)}"
        )

    # Save split indices
    indices_path = output_path / "data_split_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)
    logger.info(f"Split indices saved to {indices_path}")

    return {
        "n_min": n_min,
        "raw_counts": raw_counts,
        "balanced_counts": balanced_counts,
        "train_qa_counts": train_qa_counts,
        "balanced_data": balanced_data,
        "client_splits": client_splits,
    }
