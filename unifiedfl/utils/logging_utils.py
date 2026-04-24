from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logging(output_dir: str = "outputs/") -> logging.Logger:
    """Configure root logger to write to stdout and a log file."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("federated_qa")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def print_balancing_table(
    raw_counts: Dict[int, int],
    balanced_counts: Dict[int, int],
    train_qa_counts: Dict[int, int],
    client_names: List[str],
) -> None:
    """Print the dataset balancing summary table."""
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│               DATASET BALANCING SUMMARY                 │")
    print("├──────────────┬─────────┬──────────┬────────────────────┤")
    print("│ Client       │ Raw     │ Balanced │ QA Pairs (train)   │")
    print("├──────────────┼─────────┼──────────┼────────────────────┤")
    for cid in sorted(raw_counts.keys()):
        name = client_names[cid] if cid < len(client_names) else f"C{cid}"
        raw = raw_counts[cid]
        bal = balanced_counts[cid]
        qa = train_qa_counts[cid]
        print(f"│ {name:<12} │ {raw:>7} │ {bal:>8} │ {qa:>18} │")
    print("└──────────────┴─────────┴──────────┴────────────────────┘\n")


def print_qualitative_result(
    round_idx: int,
    client_id: int,
    model_name: str,
    context: str,
    generated: str,
) -> None:
    """Print a single qualitative evaluation result in the specified box format."""
    header = f"  QUALITATIVE EVAL — Round {round_idx} — Client {client_id} ({model_name})"
    ctx_preview = context[:200] + ("..." if len(context) > 200 else "")
    print("\n╔══════════════════════════════════════════════════════════╗")
    print(f"║{header:<58}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Context:                                               ║")
    for line in _wrap_text(ctx_preview, 54):
        print(f"║  {line:<54}  ║")
    print("║  ────────────────────────────────────────────────────   ║")
    print("║  Generated Output:                                      ║")
    for line in _wrap_text(generated, 54):
        print(f"║  {line:<54}  ║")
    print("╚══════════════════════════════════════════════════════════╝")


def print_quantitative_table(
    round_idx: int,
    results: List[Dict[str, Any]],
) -> None:
    """Print the quantitative evaluation results table."""
    print(f"\n{'═' * 56}")
    print(f"Round {round_idx} — Quantitative Evaluation Results")
    print("┌─────────────────┬──────────┬──────────┬─────────────┐")
    print("│ Client          │ ROUGE-L  │ BLEU-4   │ BERTScore F1│")
    print("├─────────────────┼──────────┼──────────┼─────────────┤")
    for row in results:
        label = row["label"]
        r = row.get("rouge_l", 0.0)
        b = row.get("bleu_4", 0.0)
        bs = row.get("bertscore_f1", 0.0)
        print(f"│ {label:<15} │  {r:.3f}   │  {b:.3f}   │    {bs:.3f}    │")
    print("└─────────────────┴──────────┴──────────┴─────────────┘")


def print_comparison_table(
    baseline: Dict[str, float],
    federated: Dict[str, float],
) -> None:
    """Print the final baseline vs federated comparison table."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              FINAL RESULTS COMPARISON                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Metric       │  Baseline  │  Federated  │     Δ        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for metric_key, label in [
        ("rouge_l", "ROUGE-L"),
        ("bleu_4", "BLEU-4"),
        ("bertscore_f1", "BERTScore F1"),
    ]:
        b_val = baseline.get(metric_key, 0.0)
        f_val = federated.get(metric_key, 0.0)
        delta = f_val - b_val
        sign = "+" if delta >= 0 else ""
        print(
            f"║  {label:<13} │   {b_val:.3f}    │    {f_val:.3f}    │  {sign}{delta:.3f}      ║"
        )
    print("╚══════════════════════════════════════════════════════════╝\n")


def _wrap_text(text: str, width: int) -> List[str]:
    """Simple word-wrap that splits text into lines of at most `width` chars."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip() if current else word
    if current:
        lines.append(current)
    return lines if lines else [""]


class JSONLogger:
    """Accumulates training metrics and writes them to a JSON file."""

    def __init__(self, output_dir: str = "outputs/") -> None:
        self._path = Path(output_dir) / "training_log.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {
            "baseline": {"metrics": {}},
            "federated": {"rounds": [], "final_metrics": {}},
        }

    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Store baseline evaluation metrics."""
        self._data["baseline"]["metrics"] = metrics
        self._flush()

    def append_round(self, round_record: Dict[str, Any]) -> None:
        """Append a federated round record."""
        self._data["federated"]["rounds"].append(round_record)
        self._flush()

    def set_federated_final(self, metrics: Dict[str, float]) -> None:
        """Store final federated evaluation metrics."""
        self._data["federated"]["final_metrics"] = metrics
        self._flush()

    def _flush(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
