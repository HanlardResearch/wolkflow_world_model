# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# 训练脚本：从 recorder 导出的 JSONL 日志训练 WorkflowWorldModel。
# 数据流：
# recorder JSONL -> WorkflowStateAdapter -> WorkflowWorldModel -> loss -> checkpoint
# 每条记录通常包含 task/state/graph/action/outcome，
# 可选包含 next_state、next_graph、returns、next_state_targets、credit_targets。

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference.policy import WorkflowStateAdapter, WorkflowWorldModel, WorkflowWorldModelConfig  # noqa: E402
from utils.file_utils import iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a workflow world model from recorder JSONL logs.")
    parser.add_argument("--data-root", type=str, default="results", help="Root directory to search for JSONL files.")
    parser.add_argument(
        "--train-data-root",
        type=str,
        default="",
        help="Root directory for training JSONL files. When set with --test-data-root, episode splitting is disabled.",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default="",
        help="Root directory for evaluation JSONL files. When set with --train-data-root, episode splitting is disabled.",
    )
    parser.add_argument(
        "--dataset-filename",
        type=str,
        default="workflow_world_model",
        help="Recorder JSONL filename prefix or exact name to collect.",
    )
    parser.add_argument("--max-files", type=int, default=-1, help="Limit the number of dataset files to read.")
    parser.add_argument("--max-records", type=int, default=-1, help="Limit the number of records to load.")
    parser.add_argument("--output-dir", type=str, default="checkpoint/workflow_world_model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.25,
        help="Loss weight for the value head.",
    )
    parser.add_argument(
        "--disable-reward-buckets",
        action="store_true",
        help="Use scalar reward regression instead of bucketed reward classification.",
    )
    parser.add_argument(
        "--reward-bucket-edges",
        type=str,
        default="-0.5,0.5",
        help="Comma-separated sorted bucket edges used when reward bucket classification is enabled.",
    )
    parser.add_argument(
        "--reward-bucket-label-smoothing",
        type=float,
        default=0.02,
        help="Label smoothing used by the reward bucket classification loss.",
    )
    parser.add_argument(
        "--disable-value-buckets",
        action="store_true",
        help="Use scalar value regression instead of bucketed value classification.",
    )
    parser.add_argument(
        "--value-bucket-edges",
        type=str,
        default="-0.75,-0.25,0.25,0.75",
        help="Comma-separated sorted bucket edges used when value bucket classification is enabled.",
    )
    parser.add_argument(
        "--value-bucket-label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing used by the value bucket classification loss.",
    )
    parser.add_argument(
        "--done-loss-weight",
        type=float,
        default=0.05,
        help="Loss weight for the done head.",
    )
    parser.add_argument(
        "--valid-loss-weight",
        type=float,
        default=0.05,
        help="Loss weight for the valid-action head.",
    )
    parser.add_argument(
        "--counterfactual-loss-weight",
        type=float,
        default=0.0,
        help="Loss weight for the counterfactual/Q head.",
    )
    parser.add_argument(
        "--kl-max-per-sample",
        type=float,
        default=5.0,
        help="Clip per-sample KL after optional latent-dimension normalization; <=0 disables clipping.",
    )
    parser.add_argument(
        "--disable-latent-target-eval",
        action="store_true",
        help="Compute next-state posterior targets in training mode instead of deterministic eval mode.",
    )
    parser.add_argument(
        "--disable-kl-dim-normalization",
        action="store_true",
        help="Optimize KL as a sum over latent dimensions instead of a per-dimension average.",
    )
    parser.add_argument(
        "--disable-latent-loss-normalization",
        action="store_true",
        help="Align latent means in raw coordinates instead of layer-normalized coordinates.",
    )
    parser.add_argument(
        "--latent-cosine-weight",
        type=float,
        default=0.25,
        help="Additional cosine-direction penalty mixed into the latent alignment loss.",
    )
    parser.add_argument(
        "--latent-logvar-weight",
        type=float,
        default=0.1,
        help="Weight for explicit prior/posterior log-variance matching.",
    )
    parser.add_argument(
        "--latent-logvar-min",
        type=float,
        default=-6.0,
        help="Lower clamp for posterior/prior latent log-variance.",
    )
    parser.add_argument(
        "--latent-logvar-max",
        type=float,
        default=2.0,
        help="Upper clamp for posterior/prior latent log-variance.",
    )
    parser.add_argument(
        "--disable-bounded-cost-output",
        action="store_true",
        help="Use an unbounded raw regression head for cost instead of sigmoid-bounded outputs.",
    )
    parser.add_argument(
        "--use-qwen-text-encoder",
        action="store_true",
        help="Use sentence-transformers Qwen text embeddings for task/evidence/step text.",
    )
    parser.add_argument(
        "--qwen-text-encoder-model-name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model name or local path used for text embeddings.",
    )
    parser.add_argument(
        "--qwen-text-encoder-batch-size",
        type=int,
        default=16,
        help="Batch size used inside the SentenceTransformer text encoder.",
    )
    parser.add_argument("--use-llm-text-encoder", action="store_true", help="Use a HuggingFace LLM to encode task/evidence text.")
    parser.add_argument(
        "--text-encoder-model-path",
        type=str,
        default="/extrahome0/HF_models/Qwen/Qwen3.5-4B",
        help="Local HuggingFace model path for the text encoder.",
    )
    parser.add_argument(
        "--text-encoder-trainable",
        action="store_true",
        help="Update the text encoder during training. Default keeps it frozen.",
    )
    parser.add_argument(
        "--text-encoder-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype used when loading the text encoder.",
    )
    parser.add_argument("--task-text-max-length", type=int, default=8192, help="Max token length for task question text.")
    parser.add_argument(
        "--evidence-text-max-length",
        type=int,
        default=128,
        help="Max token length for each evidence text item.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="workflow-world-model", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity/team.")
    parser.add_argument("--wandb-run-name", type=str, default="", help="Optional W&B run name.")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode. Use offline when network sync should be deferred.",
    )
    parser.add_argument("--use-swanlab", action="store_true", help="Log metrics to SwanLab.")
    parser.add_argument("--swanlab-project", type=str, default="workflow-world-model", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", type=str, default="", help="Optional SwanLab workspace.")
    parser.add_argument("--swanlab-run-name", type=str, default="", help="Optional SwanLab experiment name.")
    parser.add_argument("--swanlab-tags", type=str, default="", help="Comma-separated SwanLab tags.")
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="SwanLab mode. Use offline when sync should be deferred.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # 同时固定 Python 和 PyTorch 的随机种子，保证实验可复现。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_dataset_files(data_root: str, dataset_filename: str, max_files: int) -> List[str]:
    # 递归查找数据文件。既支持精确文件名，也支持按前缀匹配带时间戳的版本。
    matches: List[str] = []
    dataset_filename = str(dataset_filename or "").strip()
    filename_prefix = dataset_filename[:-6] if dataset_filename.endswith(".jsonl") else dataset_filename
    for root, _, files in os.walk(data_root):
        for file_name in files:
            if not file_name.endswith(".jsonl"):
                continue
            if dataset_filename and file_name != dataset_filename and not file_name.startswith(filename_prefix):
                continue
            matches.append(os.path.join(root, file_name))
    matches.sort()
    if max_files > 0:
        matches = matches[:max_files]
    return matches


def load_records(paths: Sequence[str], max_records: int) -> List[Dict]:
    # 读取 JSONL 记录，并在达到 max_records 后提前停止。
    records: List[Dict] = []
    for path in paths:
        file_records = iter_jsonl(path)
        if max_records > 0:
            remaining = max_records - len(records)
            if remaining <= 0:
                break
            file_records = file_records[:remaining]
        records.extend(file_records)
    return records


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_cost(value: object, clip: float = 1.0e8) -> float:
    raw = max(_safe_float(value, 0.0), 0.0)
    return min(math.log1p(raw) / math.log1p(clip), 1.0)


def _sequence_length(value: object) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    count = float(len(values))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summarize_target_distribution(values: Sequence[float], kind: str) -> Dict[str, float | str]:
    if not values:
        return {
            "kind": kind,
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "unique_count": 0.0,
            "unique_ratio": 0.0,
            "zero_rate": 0.0,
            "dominant_ratio": 0.0,
            "positive_rate": 0.0,
        }
    summary = _summarize_numeric(values)
    sorted_values = sorted(float(value) for value in values)
    rounded_values = [round(value, 6) for value in sorted_values]
    counter = Counter(rounded_values)
    count = float(len(values))
    summary.update(
        {
            "kind": kind,
            "p25": _quantile(sorted_values, 0.25),
            "p50": _quantile(sorted_values, 0.50),
            "p75": _quantile(sorted_values, 0.75),
            "unique_count": float(len(counter)),
            "unique_ratio": float(len(counter)) / count,
            "zero_rate": sum(1.0 for value in sorted_values if abs(value) <= 1.0e-8) / count,
            "dominant_ratio": max(counter.values()) / count,
            "positive_rate": sum(1.0 for value in sorted_values if value > 0.0) / count,
        }
    )
    return summary


def _top_counts(counter: Counter, limit: int = 8) -> List[Dict[str, object]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def summarize_split_records(split_name: str, records: Sequence[Dict], files: Sequence[str]) -> Dict[str, object]:
    episode_lengths: Dict[str, int] = {}
    action_counts: Counter = Counter()
    workflow_state_counts: Counter = Counter()
    task_type_counts: Counter = Counter()
    reward_values: List[float] = []
    value_values: List[float] = []
    cost_raw_values: List[float] = []
    cost_target_values: List[float] = []
    uncertainty_values: List[float] = []
    done_values: List[float] = []
    valid_action_counts: List[float] = []
    executed_step_counts: List[float] = []
    evidence_counts: List[float] = []
    graph_node_counts: List[float] = []
    next_state_present = 0.0
    next_graph_present = 0.0
    returns_present = 0.0
    next_targets_present = 0.0

    for record in records:
        episode_id = str(record.get("episode_id", "unknown"))
        episode_lengths[episode_id] = episode_lengths.get(episode_id, 0) + 1

        action = record.get("action", {})
        action_name = str(action.get("name", "unknown")).strip() or "unknown"
        action_counts[action_name] += 1

        task = record.get("task", {})
        task_type = str(task.get("task_type", task.get("type", "unknown"))).strip() or "unknown"
        task_type_counts[task_type] += 1

        state = record.get("state", {})
        workflow_state = str(state.get("workflow_state", state.get("state", "unknown"))).strip() or "unknown"
        workflow_state_counts[workflow_state] += 1

        outcome = record.get("outcome", {})
        next_state_targets = record.get("next_state_targets", {})
        returns = record.get("returns", {})

        reward = _safe_float(outcome.get("reward", 0.0))
        reward_values.append(reward)
        value_values.append(_safe_float(returns.get("mc_return", reward)))
        raw_cost = max(_safe_float(outcome.get("cost_delta", 0.0)), 0.0)
        cost_raw_values.append(raw_cost)
        cost_target_values.append(_normalize_cost(raw_cost))
        uncertainty_values.append(_safe_float(next_state_targets.get("conflict_score", 0.0)))
        done_values.append(1.0 if bool(outcome.get("done", False)) else 0.0)

        valid_actions = next_state_targets.get("valid_action_mask", state.get("valid_actions", []))
        valid_action_counts.append(float(_sequence_length(valid_actions)))
        executed_step_counts.append(float(_sequence_length(state.get("executed_steps", []))))
        evidence_counts.append(
            float(
                _sequence_length(state.get("reasoning_results", []))
                + _sequence_length(state.get("tool_results", []))
                + _sequence_length(state.get("recent_answers", []))
            )
        )
        graph_node_counts.append(float(_sequence_length(record.get("graph", {}).get("nodes", []))))

        if isinstance(record.get("next_state"), dict) and bool(record.get("next_state")):
            next_state_present += 1.0
        if isinstance(record.get("next_graph"), dict) and bool(record.get("next_graph")):
            next_graph_present += 1.0
        if isinstance(returns, dict) and bool(returns):
            returns_present += 1.0
        if isinstance(next_state_targets, dict) and bool(next_state_targets):
            next_targets_present += 1.0

    record_count = len(records)
    denom = float(record_count) if record_count > 0 else 1.0
    episode_stats = _summarize_numeric(list(episode_lengths.values()))
    return {
        "split": split_name,
        "file_count": len(files),
        "record_count": record_count,
        "episode_count": len(episode_lengths),
        "episode_length": episode_stats,
        "reward": _summarize_numeric(reward_values),
        "value": _summarize_numeric(value_values),
        "cost_delta_raw": _summarize_numeric(cost_raw_values),
        "cost_target": _summarize_numeric(cost_target_values),
        "uncertainty": _summarize_numeric(uncertainty_values),
        "done_rate": sum(done_values) / denom,
        "valid_action_count": _summarize_numeric(valid_action_counts),
        "executed_step_count": _summarize_numeric(executed_step_counts),
        "evidence_count": _summarize_numeric(evidence_counts),
        "graph_node_count": _summarize_numeric(graph_node_counts),
        "next_state_present_rate": next_state_present / denom,
        "next_graph_present_rate": next_graph_present / denom,
        "returns_present_rate": returns_present / denom,
        "next_targets_present_rate": next_targets_present / denom,
        "unique_action_count": len(action_counts),
        "unique_workflow_state_count": len(workflow_state_counts),
        "unique_task_type_count": len(task_type_counts),
        "action_vocab": sorted(action_counts.keys()),
        "workflow_state_vocab": sorted(workflow_state_counts.keys()),
        "task_type_vocab": sorted(task_type_counts.keys()),
        "top_actions": _top_counts(action_counts),
        "top_workflow_states": _top_counts(workflow_state_counts),
        "top_task_types": _top_counts(task_type_counts),
    }


def build_target_diagnostics(train_records: Sequence[Dict], val_records: Sequence[Dict]) -> Dict[str, object]:
    target_extractors = {
        "reward": ("regression", lambda record: _safe_float(record.get("outcome", {}).get("reward", 0.0))),
        "value": (
            "regression",
            lambda record: _safe_float(
                record.get("returns", {}).get("mc_return", record.get("outcome", {}).get("reward", 0.0))
            ),
        ),
        "cost_target": ("regression", lambda record: _normalize_cost(record.get("outcome", {}).get("cost_delta", 0.0))),
        "uncertainty": (
            "regression",
            lambda record: _safe_float(record.get("next_state_targets", {}).get("conflict_score", 0.0)),
        ),
        "counterfactual": (
            "regression",
            lambda record: _safe_float(
                record.get("credit_targets", {}).get("leave_one_out_gap", record.get("outcome", {}).get("reward", 0.0))
            ),
        ),
        "done": ("binary", lambda record: 1.0 if bool(record.get("outcome", {}).get("done", False)) else 0.0),
        "valid_action_count": (
            "count",
            lambda record: float(
                _sequence_length(
                    record.get("next_state_targets", {}).get(
                        "valid_action_mask",
                        record.get("state", {}).get("valid_actions", []),
                    )
                )
            ),
        ),
    }

    diagnostics: Dict[str, object] = {}
    warnings: List[str] = []
    split_records = {"train": list(train_records), "val": list(val_records)}

    for target_name, (kind, extractor) in target_extractors.items():
        split_stats: Dict[str, Dict[str, float | str]] = {}
        for split_name, records in split_records.items():
            values = [float(extractor(record)) for record in records]
            split_stats[split_name] = _summarize_target_distribution(values, kind)

        train_stats = split_stats["train"]
        val_stats = split_stats["val"]
        mean_gap = float(val_stats["mean"]) - float(train_stats["mean"])
        std_gap = float(val_stats["std"]) - float(train_stats["std"])
        diagnostics[target_name] = {
            "kind": kind,
            "train": train_stats,
            "val": val_stats,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
        }

        for split_name, stats in split_stats.items():
            split_label = f"{target_name}/{split_name}"
            if float(stats["count"]) <= 1.0:
                warnings.append(f"{split_label} has too few samples for reliable learning diagnostics.")
                continue
            if kind == "regression":
                if float(stats["std"]) < 1.0e-4:
                    warnings.append(f"{split_label} has near-zero variance; this head is effectively a constant target.")
                if float(stats["dominant_ratio"]) > 0.95:
                    warnings.append(f"{split_label} is dominated by one value ({float(stats['dominant_ratio']):.4f}).")
                if float(stats["unique_ratio"]) < 0.05:
                    warnings.append(f"{split_label} has very low label diversity ({float(stats['unique_ratio']):.4f}).")
            if kind == "binary":
                positive_rate = float(stats["positive_rate"])
                if positive_rate < 0.05 or positive_rate > 0.95:
                    warnings.append(f"{split_label} is highly imbalanced (positive_rate={positive_rate:.4f}).")

        if kind in {"regression", "count"} and abs(mean_gap) > max(0.05, 0.5 * float(train_stats["std"]) + 1.0e-6):
            warnings.append(f"{target_name} mean differs noticeably between train and val (gap={mean_gap:.4f}).")
        if kind == "binary":
            train_rate = float(train_stats["positive_rate"])
            val_rate = float(val_stats["positive_rate"])
            if abs(val_rate - train_rate) > 0.10:
                warnings.append(f"{target_name} positive rate differs noticeably between train and val (gap={val_rate - train_rate:.4f}).")

    return {"targets": diagnostics, "warnings": warnings}


def _conflict_signature(record: Dict) -> Tuple[str, str]:
    state = record.get("state", {})
    action = record.get("action", {})
    workflow_state = str(state.get("workflow_state", state.get("state", "unknown"))).strip() or "unknown"
    action_name = str(action.get("name", "unknown")).strip() or "unknown"
    return workflow_state, action_name


def _summarize_conflict_stats(records: Sequence[Dict], label_name: str) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    extractor = (
        lambda record: _safe_float(record.get("outcome", {}).get("reward", 0.0))
        if label_name == "reward"
        else _safe_float(record.get("returns", {}).get("mc_return", record.get("outcome", {}).get("reward", 0.0)))
    )
    for record in records:
        grouped.setdefault(_conflict_signature(record), []).append(float(extractor(record)))

    total_groups = 0
    conflicting_groups = 0
    conflicting_records = 0
    examples: List[Dict[str, object]] = []
    for (workflow_state, action_name), values in grouped.items():
        unique_values = sorted({round(value, 6) for value in values})
        if len(values) <= 1:
            continue
        total_groups += 1
        if len(unique_values) <= 1:
            continue
        conflicting_groups += 1
        conflicting_records += len(values)
        examples.append(
            {
                "workflow_state": workflow_state,
                "action_name": action_name,
                "count": len(values),
                "unique_values": unique_values[:8],
                "min": min(values),
                "max": max(values),
            }
        )
    examples.sort(key=lambda item: (-int(item["count"]), str(item["action_name"]), str(item["workflow_state"])))
    record_count = len(records)
    return {
        "group_count": total_groups,
        "conflicting_group_count": conflicting_groups,
        "conflicting_group_ratio": conflicting_groups / max(total_groups, 1),
        "conflicting_record_count": conflicting_records,
        "conflicting_record_ratio": conflicting_records / max(record_count, 1),
        "examples": examples[:10],
    }


def build_conflict_diagnostics(train_records: Sequence[Dict], val_records: Sequence[Dict]) -> Dict[str, object]:
    diagnostics = {
        "reward": {
            "train": _summarize_conflict_stats(train_records, "reward"),
            "val": _summarize_conflict_stats(val_records, "reward"),
        },
        "value": {
            "train": _summarize_conflict_stats(train_records, "value"),
            "val": _summarize_conflict_stats(val_records, "value"),
        },
    }
    warnings: List[str] = []
    for label_name, split_reports in diagnostics.items():
        for split_name, report in split_reports.items():
            conflicting_ratio = float(report["conflicting_record_ratio"])
            group_ratio = float(report["conflicting_group_ratio"])
            if conflicting_ratio > 0.05:
                warnings.append(
                    f"{label_name}/{split_name} has conflicting labels for repeated state-action signatures "
                    f"(record_ratio={conflicting_ratio:.4f}, group_ratio={group_ratio:.4f})."
                )
    return {"labels": diagnostics, "warnings": warnings}


def build_dataset_report(
    train_records: Sequence[Dict],
    val_records: Sequence[Dict],
    train_files: Sequence[str],
    val_files: Sequence[str],
) -> Dict[str, object]:
    train_summary = summarize_split_records("train", train_records, train_files)
    val_summary = summarize_split_records("val", val_records, val_files)
    target_diagnostics = build_target_diagnostics(train_records, val_records)
    conflict_diagnostics = build_conflict_diagnostics(train_records, val_records)

    train_actions = set(train_summary["action_vocab"])
    val_actions = set(val_summary["action_vocab"])
    unseen_val_actions = sorted(val_actions - train_actions)
    shared_actions = sorted(train_actions & val_actions)
    val_action_coverage = len(shared_actions) / max(len(val_actions), 1)

    comparison = {
        "reward_mean_gap": val_summary["reward"]["mean"] - train_summary["reward"]["mean"],
        "value_mean_gap": val_summary["value"]["mean"] - train_summary["value"]["mean"],
        "cost_target_mean_gap": val_summary["cost_target"]["mean"] - train_summary["cost_target"]["mean"],
        "uncertainty_mean_gap": val_summary["uncertainty"]["mean"] - train_summary["uncertainty"]["mean"],
        "done_rate_gap": val_summary["done_rate"] - train_summary["done_rate"],
        "valid_action_count_gap": val_summary["valid_action_count"]["mean"] - train_summary["valid_action_count"]["mean"],
        "val_action_coverage": val_action_coverage,
        "shared_action_count": len(shared_actions),
        "unseen_val_actions": unseen_val_actions,
    }

    warnings: List[str] = []
    if train_summary["record_count"] < 100:
        warnings.append(f"Training split is very small: {train_summary['record_count']} records.")
    if train_summary["episode_count"] < 10:
        warnings.append(f"Training split has only {train_summary['episode_count']} episodes.")
    if val_summary["record_count"] and val_summary["episode_count"] < 5:
        warnings.append(f"Validation split has only {val_summary['episode_count']} episodes.")
    if abs(comparison["reward_mean_gap"]) > 0.25:
        warnings.append(
            "Reward distribution differs noticeably between train and val "
            f"(gap={comparison['reward_mean_gap']:.4f})."
        )
    if abs(comparison["value_mean_gap"]) > 0.25:
        warnings.append(
            "Value distribution differs noticeably between train and val "
            f"(gap={comparison['value_mean_gap']:.4f})."
        )
    if abs(comparison["uncertainty_mean_gap"]) > 0.05:
        warnings.append(
            "Uncertainty/conflict targets differ noticeably between train and val "
            f"(gap={comparison['uncertainty_mean_gap']:.4f})."
        )
    if abs(comparison["done_rate_gap"]) > 0.15:
        warnings.append(f"Done rate differs noticeably between train and val (gap={comparison['done_rate_gap']:.4f}).")
    if unseen_val_actions:
        warnings.append(
            "Validation includes actions not seen in training: " + ", ".join(unseen_val_actions[:8])
        )
    comparison["warnings"] = warnings

    return {
        "overview": {
            "train_file_count": len(train_files),
            "val_file_count": len(val_files),
            "train_record_count": len(train_records),
            "val_record_count": len(val_records),
        },
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
        "comparison": comparison,
        "target_diagnostics": target_diagnostics,
        "conflict_diagnostics": conflict_diagnostics,
    }


def _format_report_value(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "nan"


def render_dataset_report_markdown(report: Dict[str, object]) -> str:
    overview = report["overview"]
    comparison = report["comparison"]
    target_diagnostics = report.get("target_diagnostics", {})
    conflict_diagnostics = report.get("conflict_diagnostics", {})
    lines = [
        "# Dataset Split Report",
        "",
        "## Overview",
        "",
        f"- Train files: {overview['train_file_count']}",
        f"- Validation files: {overview['val_file_count']}",
        f"- Train records: {overview['train_record_count']}",
        f"- Validation records: {overview['val_record_count']}",
        "",
    ]
    for split_name in ("train", "val"):
        summary = report["splits"][split_name]
        reward = summary["reward"]
        value = summary["value"]
        uncertainty = summary["uncertainty"]
        valid_actions = summary["valid_action_count"]
        episode_length = summary["episode_length"]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- Episodes: {summary['episode_count']}",
                (
                    "- Episode length mean/std/min/max: "
                    f"{_format_report_value(episode_length['mean'])} / "
                    f"{_format_report_value(episode_length['std'])} / "
                    f"{_format_report_value(episode_length['min'])} / "
                    f"{_format_report_value(episode_length['max'])}"
                ),
                (
                    "- Reward mean/std/min/max: "
                    f"{_format_report_value(reward['mean'])} / "
                    f"{_format_report_value(reward['std'])} / "
                    f"{_format_report_value(reward['min'])} / "
                    f"{_format_report_value(reward['max'])}"
                ),
                (
                    "- Value mean/std/min/max: "
                    f"{_format_report_value(value['mean'])} / "
                    f"{_format_report_value(value['std'])} / "
                    f"{_format_report_value(value['min'])} / "
                    f"{_format_report_value(value['max'])}"
                ),
                (
                    "- Uncertainty mean/std: "
                    f"{_format_report_value(uncertainty['mean'])} / "
                    f"{_format_report_value(uncertainty['std'])}"
                ),
                (
                    "- Valid action count mean/std: "
                    f"{_format_report_value(valid_actions['mean'])} / "
                    f"{_format_report_value(valid_actions['std'])}"
                ),
                f"- Done rate: {_format_report_value(summary['done_rate'])}",
                f"- Next-state present rate: {_format_report_value(summary['next_state_present_rate'])}",
                f"- Next-graph present rate: {_format_report_value(summary['next_graph_present_rate'])}",
                f"- Returns present rate: {_format_report_value(summary['returns_present_rate'])}",
                f"- Next-targets present rate: {_format_report_value(summary['next_targets_present_rate'])}",
                "- Top actions: "
                + ", ".join(f"{item['name']}({item['count']})" for item in summary["top_actions"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Comparison",
            "",
            f"- Reward mean gap (val-train): {_format_report_value(comparison['reward_mean_gap'])}",
            f"- Value mean gap (val-train): {_format_report_value(comparison['value_mean_gap'])}",
            f"- Cost target mean gap (val-train): {_format_report_value(comparison['cost_target_mean_gap'])}",
            f"- Uncertainty mean gap (val-train): {_format_report_value(comparison['uncertainty_mean_gap'])}",
            f"- Done rate gap (val-train): {_format_report_value(comparison['done_rate_gap'])}",
            f"- Validation action coverage by training action vocab: {_format_report_value(comparison['val_action_coverage'])}",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = comparison["warnings"] or ["No heuristic issues detected."]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")

    if target_diagnostics:
        lines.extend(
            [
                "## Target Diagnostics",
                "",
            ]
        )
        for target_name, target_report in target_diagnostics.get("targets", {}).items():
            train_stats = target_report["train"]
            val_stats = target_report["val"]
            lines.extend(
                [
                    f"### {target_name}",
                    "",
                    f"- Kind: {target_report['kind']}",
                    (
                        "- Train mean/std/p50/unique_ratio/dominant_ratio: "
                        f"{_format_report_value(float(train_stats['mean']))} / "
                        f"{_format_report_value(float(train_stats['std']))} / "
                        f"{_format_report_value(float(train_stats['p50']))} / "
                        f"{_format_report_value(float(train_stats['unique_ratio']))} / "
                        f"{_format_report_value(float(train_stats['dominant_ratio']))}"
                    ),
                    (
                        "- Val mean/std/p50/unique_ratio/dominant_ratio: "
                        f"{_format_report_value(float(val_stats['mean']))} / "
                        f"{_format_report_value(float(val_stats['std']))} / "
                        f"{_format_report_value(float(val_stats['p50']))} / "
                        f"{_format_report_value(float(val_stats['unique_ratio']))} / "
                        f"{_format_report_value(float(val_stats['dominant_ratio']))}"
                    ),
                    f"- Mean gap (val-train): {_format_report_value(float(target_report['mean_gap']))}",
                    "",
                ]
            )
        lines.extend(["### Target Warnings", ""])
        target_warnings = target_diagnostics.get("warnings", []) or ["No target-level issues detected."]
        lines.extend(f"- {warning}" for warning in target_warnings)
        lines.append("")
    if conflict_diagnostics:
        lines.extend(["## Label Conflicts", ""])
        for label_name, split_reports in conflict_diagnostics.get("labels", {}).items():
            lines.append(f"### {label_name}")
            lines.append("")
            for split_name in ("train", "val"):
                report_item = split_reports.get(split_name, {})
                lines.append(
                    f"- {split_name}: conflicting groups={int(report_item.get('conflicting_group_count', 0))}/"
                    f"{int(report_item.get('group_count', 0))} "
                    f"({_format_report_value(float(report_item.get('conflicting_group_ratio', 0.0)))})"
                )
                lines.append(
                    f"- {split_name}: conflicting records={int(report_item.get('conflicting_record_count', 0))} "
                    f"({_format_report_value(float(report_item.get('conflicting_record_ratio', 0.0)))})"
                )
            examples = split_reports.get("val", {}).get("examples", []) or split_reports.get("train", {}).get("examples", [])
            if examples:
                lines.append("- Example signatures:")
                for example in examples[:5]:
                    lines.append(
                        f"  - {example['action_name']} @ {example['workflow_state']} "
                        f"count={example['count']} values={example['unique_values']}"
                    )
            lines.append("")
        lines.extend(["### Conflict Warnings", ""])
        conflict_warnings = conflict_diagnostics.get("warnings", []) or ["No repeated-signature label conflicts detected."]
        lines.extend(f"- {warning}" for warning in conflict_warnings)
        lines.append("")
    return "\n".join(lines)


def write_dataset_report(output_dir: str, report: Dict[str, object]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "dataset_report.json")
    md_path = os.path.join(output_dir, "dataset_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_dataset_report_markdown(report))
    return {"json": json_path, "markdown": md_path}


def print_dataset_report_summary(report: Dict[str, object], report_paths: Dict[str, str]) -> None:
    overview = report["overview"]
    comparison = report["comparison"]
    target_diagnostics = report.get("target_diagnostics", {})
    conflict_diagnostics = report.get("conflict_diagnostics", {})
    print(
        "[Data] "
        f"train_records={overview['train_record_count']} "
        f"val_records={overview['val_record_count']} "
        f"train_files={overview['train_file_count']} "
        f"val_files={overview['val_file_count']}"
    )
    print(
        "[Data] "
        f"reward_gap={comparison['reward_mean_gap']:.4f} "
        f"value_gap={comparison['value_mean_gap']:.4f} "
        f"uncertainty_gap={comparison['uncertainty_mean_gap']:.4f} "
        f"action_coverage={comparison['val_action_coverage']:.4f}"
    )
    for warning in comparison["warnings"]:
        print(f"[Data Warning] {warning}")
    for warning in target_diagnostics.get("warnings", [])[:8]:
        print(f"[Target Warning] {warning}")
    for warning in conflict_diagnostics.get("warnings", [])[:8]:
        print(f"[Conflict Warning] {warning}")
    print(f"[Data] report_json={report_paths['json']}")
    print(f"[Data] report_markdown={report_paths['markdown']}")


def collect_vocab(records: Sequence[Dict]) -> Tuple[List[str], List[str]]:
    # 从图结构、动作记录和状态字段中收集角色与动作词表。
    roles = set()
    actions = set()
    for record in records:
        for role_name in record.get("graph", {}).get("nodes", []):
            roles.add(str(role_name))
        action = record.get("action", {})
        if action.get("name"):
            actions.add(str(action["name"]))
        state = record.get("state", {})
        for step in state.get("executed_steps", []):
            if step.get("agent"):
                roles.add(str(step["agent"]))
            if step.get("action"):
                actions.add(str(step["action"]))
        for action_name in record.get("next_state_targets", {}).get("valid_action_mask", state.get("valid_actions", [])):
            actions.add(str(action_name))
    if not roles:
        roles.add("TerminatorAgent")
    if not actions:
        actions.update(roles)
    return sorted(roles), sorted(actions)


def split_by_episode(records: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    # 按 episode 切分训练集和验证集，避免同一条轨迹同时出现在两个集合中。
    episode_to_records: Dict[str, List[Dict]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", "unknown"))
        episode_to_records.setdefault(episode_id, []).append(record)
    episode_ids = list(episode_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    val_count = int(len(episode_ids) * val_ratio)
    val_episodes = set(episode_ids[:val_count])
    train_records: List[Dict] = []
    val_records: List[Dict] = []
    for episode_id, episode_records in episode_to_records.items():
        if episode_id in val_episodes:
            val_records.extend(episode_records)
        else:
            train_records.extend(episode_records)
    if not train_records:
        train_records = list(records)
        val_records = []
    return train_records, val_records


def build_next_state_records(records: Sequence[Dict]) -> List[Dict]:
    # 将 next_state/next_graph 重组为“下一时刻观测”，用于监督 latent transition。
    next_records: List[Dict] = []
    for record in records:
        next_record = dict(record)
        next_record["graph"] = record.get("next_graph", record.get("graph", {}))
        next_record["state"] = record.get("next_state", record.get("state", {}))
        next_records.append(next_record)
    return next_records


def batched(records: Sequence[Dict], batch_size: int, shuffle: bool) -> Iterable[List[Dict]]:
    # 轻量级 batch 生成器，避免引入额外 DataLoader 依赖。
    indices = list(range(len(records)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [records[index] for index in batch_indices]


def group_records_by_episode(records: Sequence[Dict]) -> List[List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", "unknown"))
        grouped.setdefault(episode_id, []).append(record)
    episodes = []
    for episode_records in grouped.values():
        episodes.append(
            sorted(
                episode_records,
                key=lambda item: (
                    int(item.get("t", 0) or 0),
                    str(item.get("path_id", "")),
                ),
            )
        )
    return episodes


def batched_episodes(episodes: Sequence[Sequence[Dict]], batch_size: int, shuffle: bool) -> Iterable[List[List[Dict]]]:
    episode_list = [list(episode) for episode in episodes if episode]
    if shuffle:
        random.shuffle(episode_list)
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(episode_list), batch_size):
        yield episode_list[start : start + batch_size]


def build_adapter_config(args: argparse.Namespace) -> WorkflowWorldModelConfig:
    config = WorkflowWorldModelConfig()
    reward_bucket_edges = [float(part.strip()) for part in str(args.reward_bucket_edges).split(",") if part.strip()]
    bucket_edges = [float(part.strip()) for part in str(args.value_bucket_edges).split(",") if part.strip()]
    config.use_qwen_text_encoder = bool(args.use_qwen_text_encoder)
    config.qwen_text_encoder_model_name = str(args.qwen_text_encoder_model_name)
    config.qwen_text_encoder_batch_size = int(args.qwen_text_encoder_batch_size)
    config.use_llm_text_encoder = bool(args.use_llm_text_encoder)
    if config.use_qwen_text_encoder:
        config.use_llm_text_encoder = False
    config.text_encoder_model_path = str(args.text_encoder_model_path)
    config.text_encoder_freeze = not bool(args.text_encoder_trainable)
    config.text_encoder_dtype = str(args.text_encoder_dtype)
    config.task_text_max_length = int(args.task_text_max_length)
    config.evidence_text_max_length = int(args.evidence_text_max_length)
    config.stable_next_posterior_targets = not bool(args.disable_latent_target_eval)
    config.normalize_kl_by_latent_dim = not bool(args.disable_kl_dim_normalization)
    config.max_kl_per_sample = float(args.kl_max_per_sample)
    config.normalize_latent_alignment = not bool(args.disable_latent_loss_normalization)
    config.latent_cosine_weight = float(args.latent_cosine_weight)
    config.latent_logvar_weight = float(args.latent_logvar_weight)
    config.latent_logvar_min = float(args.latent_logvar_min)
    config.latent_logvar_max = float(args.latent_logvar_max)
    config.bound_cost_output = not bool(args.disable_bounded_cost_output)
    config.use_reward_buckets = not bool(args.disable_reward_buckets)
    config.reward_bucket_edges = tuple(reward_bucket_edges) if reward_bucket_edges else tuple(config.reward_bucket_edges)
    config.reward_bucket_label_smoothing = float(args.reward_bucket_label_smoothing)
    config.use_value_buckets = not bool(args.disable_value_buckets)
    config.value_bucket_edges = tuple(bucket_edges) if bucket_edges else tuple(config.value_bucket_edges)
    config.value_bucket_label_smoothing = float(args.value_bucket_label_smoothing)
    config.loss_weights["value"] = float(args.value_loss_weight)
    config.loss_weights["done"] = float(args.done_loss_weight)
    config.loss_weights["valid"] = float(args.valid_loss_weight)
    config.loss_weights["counterfactual"] = float(args.counterfactual_loss_weight)
    return config


def build_model_config(adapter: WorkflowStateAdapter) -> WorkflowWorldModelConfig:
    # 根据 adapter 扫描到的词表规模修正 embedding 配置。
    config = replace(adapter.config)
    config.num_roles = max(len(adapter.role_to_id), 1)
    config.num_actions = max(len(adapter.action_to_id), 1)
    config.num_task_types = max(len(adapter.task_type_to_id), 1)
    config.num_workflow_states = max(len(adapter.workflow_state_to_id), 1)
    return config


def _build_tracker_config(
    args: argparse.Namespace,
    model_config: WorkflowWorldModelConfig,
    dataset_size: int,
) -> Dict[str, Any]:
    return {
        "data_root": args.data_root,
        "train_data_root": args.train_data_root,
        "test_data_root": args.test_data_root,
        "dataset_filename": args.dataset_filename,
        "max_files": args.max_files,
        "max_records": args.max_records,
        "dataset_size": dataset_size,
        "output_dir": args.output_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "gradient_clip": args.gradient_clip,
        "kl_max_per_sample": args.kl_max_per_sample,
        "disable_latent_target_eval": args.disable_latent_target_eval,
        "disable_kl_dim_normalization": args.disable_kl_dim_normalization,
        "disable_latent_loss_normalization": args.disable_latent_loss_normalization,
        "latent_cosine_weight": args.latent_cosine_weight,
        "latent_logvar_weight": args.latent_logvar_weight,
        "latent_logvar_min": args.latent_logvar_min,
        "latent_logvar_max": args.latent_logvar_max,
        "use_qwen_text_encoder": args.use_qwen_text_encoder,
        "qwen_text_encoder_model_name": args.qwen_text_encoder_model_name,
        "qwen_text_encoder_batch_size": args.qwen_text_encoder_batch_size,
        "use_llm_text_encoder": args.use_llm_text_encoder,
        "text_encoder_model_path": args.text_encoder_model_path,
        "text_encoder_trainable": args.text_encoder_trainable,
        "text_encoder_dtype": args.text_encoder_dtype,
        "task_text_max_length": args.task_text_max_length,
        "evidence_text_max_length": args.evidence_text_max_length,
        "model_config": {
            "task_dim": model_config.task_dim,
            "step_dim": model_config.step_dim,
            "evidence_dim": model_config.evidence_dim,
            "budget_dim": model_config.budget_dim,
            "node_dim": model_config.node_dim,
            "action_dim": model_config.action_dim,
            "embed_dim": model_config.embed_dim,
            "model_dim": model_config.model_dim,
            "hidden_dim": model_config.hidden_dim,
            "latent_dim": model_config.latent_dim,
            "dropout": model_config.dropout,
            "num_heads": model_config.num_heads,
            "num_layers": model_config.num_layers,
            "max_steps": model_config.max_steps,
            "max_evidence": model_config.max_evidence,
            "max_nodes": model_config.max_nodes,
            "num_roles": model_config.num_roles,
            "num_actions": model_config.num_actions,
            "num_task_types": model_config.num_task_types,
            "num_workflow_states": model_config.num_workflow_states,
            "use_qwen_text_encoder": model_config.use_qwen_text_encoder,
            "qwen_text_encoder_model_name": model_config.qwen_text_encoder_model_name,
            "qwen_text_encoder_batch_size": model_config.qwen_text_encoder_batch_size,
            "use_llm_text_encoder": model_config.use_llm_text_encoder,
            "text_encoder_model_path": model_config.text_encoder_model_path,
            "text_encoder_freeze": model_config.text_encoder_freeze,
            "text_encoder_dtype": model_config.text_encoder_dtype,
            "task_text_max_length": model_config.task_text_max_length,
            "evidence_text_max_length": model_config.evidence_text_max_length,
            "stable_next_posterior_targets": model_config.stable_next_posterior_targets,
            "normalize_kl_by_latent_dim": model_config.normalize_kl_by_latent_dim,
            "max_kl_per_sample": model_config.max_kl_per_sample,
            "normalize_latent_alignment": model_config.normalize_latent_alignment,
            "latent_cosine_weight": model_config.latent_cosine_weight,
            "latent_logvar_weight": model_config.latent_logvar_weight,
            "latent_logvar_min": model_config.latent_logvar_min,
        "latent_logvar_max": model_config.latent_logvar_max,
        "bound_cost_output": model_config.bound_cost_output,
        "use_reward_buckets": model_config.use_reward_buckets,
        "reward_bucket_edges": list(model_config.reward_bucket_edges),
        "reward_bucket_label_smoothing": model_config.reward_bucket_label_smoothing,
        "use_value_buckets": model_config.use_value_buckets,
        "value_bucket_edges": list(model_config.value_bucket_edges),
        "value_bucket_label_smoothing": model_config.value_bucket_label_smoothing,
        "value_loss_weight": model_config.loss_weights.get("value", 0.0),
        "done_loss_weight": model_config.loss_weights.get("done", 0.0),
        "valid_loss_weight": model_config.loss_weights.get("valid", 0.0),
        "counterfactual_loss_weight": model_config.loss_weights.get("counterfactual", 0.0),
        "aux_names": list(model_config.aux_names),
        "loss_weights": dict(model_config.loss_weights),
        },
    }


def resolve_dataset_splits(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str], List[Dict], List[Dict]]:
    train_root = str(args.train_data_root or "").strip()
    test_root = str(args.test_data_root or "").strip()

    if bool(train_root) != bool(test_root):
        raise ValueError("--train-data-root and --test-data-root must be provided together.")

    if train_root and test_root:
        train_files = find_dataset_files(train_root, args.dataset_filename, args.max_files)
        if not train_files:
            raise FileNotFoundError(
                f"No training dataset JSONL files matching '{args.dataset_filename}' were found under '{train_root}'."
            )
        test_files = find_dataset_files(test_root, args.dataset_filename, args.max_files)
        if not test_files:
            raise FileNotFoundError(
                f"No evaluation dataset JSONL files matching '{args.dataset_filename}' were found under '{test_root}'."
            )

        train_records = load_records(train_files, args.max_records)
        test_records = load_records(test_files, args.max_records)
        if not train_records:
            raise ValueError("Training dataset files were found, but no training records could be loaded.")
        if not test_records:
            raise ValueError("Evaluation dataset files were found, but no evaluation records could be loaded.")

        all_files = train_files + test_files
        all_records = train_records + test_records
        return all_files, train_files, test_files, train_records, test_records

    dataset_files = find_dataset_files(args.data_root, args.dataset_filename, args.max_files)
    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset JSONL files matching '{args.dataset_filename}' were found under '{args.data_root}'."
        )

    records = load_records(dataset_files, args.max_records)
    if not records:
        raise ValueError("Dataset files were found, but no records could be loaded.")

    train_records, test_records = split_by_episode(records, args.val_ratio, args.seed)
    return dataset_files, dataset_files, [], train_records, test_records


class ExperimentTracker:
    def __init__(self, backend: str, client: Any, run: Any) -> None:
        self.backend = backend
        self.client = client
        self.run = run

    def log(self, payload: Dict[str, float], step: int) -> None:
        if self.backend == "wandb":
            self.run.log(payload, step=step)
            return
        if hasattr(self.run, "log"):
            try:
                self.run.log(payload, step=step)
                return
            except TypeError:
                self.run.log(payload)
                return
        if hasattr(self.client, "log"):
            try:
                self.client.log(payload, step=step)
                return
            except TypeError:
                self.client.log(payload)
                return
        raise AttributeError(f"{self.backend} logger does not provide a usable log method.")

    def finish(self) -> None:
        if hasattr(self.run, "finish"):
            self.run.finish()
            return
        if hasattr(self.client, "finish"):
            self.client.finish()


def setup_tracking(args: argparse.Namespace, model_config: WorkflowWorldModelConfig, dataset_size: int) -> Optional[ExperimentTracker]:
    if args.use_wandb and args.use_swanlab:
        raise ValueError("Choose only one tracker: --use-wandb or --use-swanlab.")
    tracker_config = _build_tracker_config(args, model_config, dataset_size)
    if args.use_wandb:
        if args.wandb_mode == "disabled":
            return None
        return setup_wandb(args, tracker_config)
    if args.use_swanlab:
        if args.swanlab_mode == "disabled":
            return None
        return setup_swanlab(args, tracker_config)
    return None


def setup_wandb(args: argparse.Namespace, tracker_config: Dict[str, Any]) -> ExperimentTracker:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it or run without --use-wandb.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        tags=tags or None,
        config=tracker_config,
    )
    return ExperimentTracker("wandb", wandb, run)


def setup_swanlab(args: argparse.Namespace, tracker_config: Dict[str, Any]) -> ExperimentTracker:
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("swanlab is not installed. Install it or run without --use-swanlab.") from exc

    tags = [tag.strip() for tag in args.swanlab_tags.split(",") if tag.strip()]
    init_kwargs: Dict[str, Any] = {
        "project": args.swanlab_project,
        "config": tracker_config,
    }
    if args.swanlab_run_name:
        init_kwargs["experiment_name"] = args.swanlab_run_name
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    if tags:
        init_kwargs["tags"] = tags
    if args.swanlab_mode != "online":
        init_kwargs["mode"] = args.swanlab_mode

    try:
        run = swanlab.init(**init_kwargs)
    except TypeError:
        fallback_kwargs = dict(init_kwargs)
        if "experiment_name" in fallback_kwargs:
            fallback_kwargs["name"] = fallback_kwargs.pop("experiment_name")
        run = swanlab.init(**fallback_kwargs)
    return ExperimentTracker("swanlab", swanlab, run)


def log_metrics_to_tracker(
    tracker: Optional[ExperimentTracker],
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    best_val: float,
    key_val_metrics: Optional[Dict[str, float]] = None,
) -> None:
    if tracker is None:
        return
    payload: Dict[str, float] = {"epoch": float(epoch)}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            payload[f"{split}/{name}"] = float(value)
    if key_val_metrics:
        for name, value in key_val_metrics.items():
            payload[f"key-val-metrics/{name}"] = float(value)
    payload["val/best_total"] = float(best_val)
    tracker.log(payload, step=epoch)


def _accumulate_regression_metrics(
    stats: Dict[str, Dict[str, float]],
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    pred = prediction.detach().float().reshape(-1)
    tgt = target.detach().float().reshape(-1)
    if pred.numel() == 0:
        return
    diff = pred - tgt
    bucket = stats.setdefault(name, {"kind": "regression"})
    bucket["count"] = bucket.get("count", 0.0) + float(pred.numel())
    bucket["pred_sum"] = bucket.get("pred_sum", 0.0) + float(pred.sum().item())
    bucket["target_sum"] = bucket.get("target_sum", 0.0) + float(tgt.sum().item())
    bucket["target_sq_sum"] = bucket.get("target_sq_sum", 0.0) + float(tgt.pow(2).sum().item())
    bucket["abs_sum"] = bucket.get("abs_sum", 0.0) + float(diff.abs().sum().item())
    bucket["sq_sum"] = bucket.get("sq_sum", 0.0) + float(diff.pow(2).sum().item())


def _accumulate_binary_metrics(
    stats: Dict[str, Dict[str, float]],
    name: str,
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> None:
    probs = probabilities.detach().float().reshape(-1)
    tgt = target.detach().float().reshape(-1)
    if probs.numel() == 0:
        return
    pred_label = probs >= 0.5
    true_label = tgt >= 0.5
    bucket = stats.setdefault(name, {"kind": "binary"})
    bucket["count"] = bucket.get("count", 0.0) + float(probs.numel())
    bucket["prob_sum"] = bucket.get("prob_sum", 0.0) + float(probs.sum().item())
    bucket["target_sum"] = bucket.get("target_sum", 0.0) + float(tgt.sum().item())
    bucket["correct_sum"] = bucket.get("correct_sum", 0.0) + float(pred_label.eq(true_label).float().sum().item())
    bucket["sq_sum"] = bucket.get("sq_sum", 0.0) + float((probs - tgt).pow(2).sum().item())


def _accumulate_multilabel_metrics(
    stats: Dict[str, Dict[str, float]],
    name: str,
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> None:
    probs = probabilities.detach().float()
    tgt = target.detach().float()
    if probs.numel() == 0:
        return
    pred_label = probs >= 0.5
    true_label = tgt >= 0.5
    bucket = stats.setdefault(name, {"kind": "multilabel"})
    bucket["label_count"] = bucket.get("label_count", 0.0) + float(probs.numel())
    bucket["sample_count"] = bucket.get("sample_count", 0.0) + float(probs.shape[0])
    bucket["prob_sum"] = bucket.get("prob_sum", 0.0) + float(probs.sum().item())
    bucket["target_sum"] = bucket.get("target_sum", 0.0) + float(tgt.sum().item())
    bucket["correct_sum"] = bucket.get("correct_sum", 0.0) + float(pred_label.eq(true_label).float().sum().item())
    bucket["exact_sum"] = bucket.get("exact_sum", 0.0) + float(pred_label.eq(true_label).all(dim=1).float().sum().item())
    bucket["tp"] = bucket.get("tp", 0.0) + float((pred_label & true_label).float().sum().item())
    bucket["fp"] = bucket.get("fp", 0.0) + float((pred_label & ~true_label).float().sum().item())
    bucket["fn"] = bucket.get("fn", 0.0) + float((~pred_label & true_label).float().sum().item())
    bucket["sq_sum"] = bucket.get("sq_sum", 0.0) + float((probs - tgt).pow(2).sum().item())


def _accumulate_batch_metrics(
    model: WorkflowWorldModel,
    loss_sums: Dict[str, float],
    prediction_stats: Dict[str, Dict[str, float]],
    losses: Dict[str, torch.Tensor],
    output,
    batch,
) -> None:
    for name, loss in losses.items():
        loss_sums[name] = loss_sums.get(name, 0.0) + float(loss.detach().cpu().item())

    targets = batch.targets
    if targets is None:
        return

    if targets.reward is not None:
        _accumulate_regression_metrics(prediction_stats, "reward", output.reward, targets.reward)
    if targets.cost is not None:
        _accumulate_regression_metrics(prediction_stats, "cost", output.cost, targets.cost)
    if targets.value is not None:
        _accumulate_regression_metrics(prediction_stats, "value", output.value, targets.value)
    if targets.uncertainty is not None:
        _accumulate_regression_metrics(prediction_stats, "uncertainty", output.uncertainty, targets.uncertainty)
    if targets.done is not None:
        _accumulate_binary_metrics(prediction_stats, "done", torch.sigmoid(output.done_logits), targets.done.float())
    if targets.next_valid_mask is not None:
        _accumulate_multilabel_metrics(
            prediction_stats,
            "valid",
            torch.sigmoid(output.valid_action_logits),
            targets.next_valid_mask.float(),
        )
    for name, target in targets.aux.items():
        if name in output.aux:
            _accumulate_regression_metrics(prediction_stats, f"aux_{name}", output.aux[name], target)
    if targets.counterfactual_credit is not None and float(model.config.loss_weights.get("counterfactual", 0.0)) > 0:
        _accumulate_regression_metrics(
            prediction_stats,
            "counterfactual",
            model.q_value(output),
            targets.counterfactual_credit,
        )


def _finalize_epoch_metrics(
    loss_sums: Dict[str, float],
    prediction_stats: Dict[str, Dict[str, float]],
    step_count: int,
) -> Dict[str, float]:
    metrics = {name: value / max(step_count, 1) for name, value in loss_sums.items()}
    for name, bucket in prediction_stats.items():
        kind = bucket.get("kind")
        if kind == "regression":
            count = max(bucket.get("count", 0.0), 1.0)
            target_mean = bucket.get("target_sum", 0.0) / count
            target_second_moment = bucket.get("target_sq_sum", 0.0) / count
            target_variance = max(target_second_moment - target_mean * target_mean, 0.0)
            mse = bucket.get("sq_sum", 0.0) / count
            metrics[f"{name}_pred_mean"] = bucket.get("pred_sum", 0.0) / count
            metrics[f"{name}_target_mean"] = target_mean
            metrics[f"{name}_mae"] = bucket.get("abs_sum", 0.0) / count
            metrics[f"{name}_rmse"] = math.sqrt(mse)
            if target_variance <= 1.0e-12:
                metrics[f"{name}_skill"] = 1.0 if mse <= 1.0e-12 else 0.0
            else:
                metrics[f"{name}_skill"] = max(0.0, 1.0 - mse / target_variance)
        elif kind == "binary":
            count = max(bucket.get("count", 0.0), 1.0)
            metrics[f"{name}_prob_mean"] = bucket.get("prob_sum", 0.0) / count
            metrics[f"{name}_target_mean"] = bucket.get("target_sum", 0.0) / count
            metrics[f"{name}_acc"] = bucket.get("correct_sum", 0.0) / count
            metrics[f"{name}_brier"] = bucket.get("sq_sum", 0.0) / count
        elif kind == "multilabel":
            label_count = max(bucket.get("label_count", 0.0), 1.0)
            sample_count = max(bucket.get("sample_count", 0.0), 1.0)
            tp = bucket.get("tp", 0.0)
            fp = bucket.get("fp", 0.0)
            fn = bucket.get("fn", 0.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            metrics[f"{name}_prob_mean"] = bucket.get("prob_sum", 0.0) / label_count
            metrics[f"{name}_target_density"] = bucket.get("target_sum", 0.0) / label_count
            metrics[f"{name}_label_acc"] = bucket.get("correct_sum", 0.0) / label_count
            metrics[f"{name}_exact_match"] = bucket.get("exact_sum", 0.0) / sample_count
            metrics[f"{name}_precision"] = precision
            metrics[f"{name}_recall"] = recall
            metrics[f"{name}_f1"] = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
            metrics[f"{name}_brier"] = bucket.get("sq_sum", 0.0) / label_count
    return metrics


def _format_metric(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "nan"


def _format_loss_line(metrics: Dict[str, float]) -> str:
    order = ("total", "latent", "latent_logvar", "kl", "reward", "cost", "done", "value", "uncertainty", "valid", "aux", "counterfactual")
    parts = [f"{name}={_format_metric(metrics[name])}" for name in order if name in metrics]
    return " ".join(parts)


def _format_regression_summary(metrics: Dict[str, float], name: str) -> str:
    key = f"{name}_mae"
    if key not in metrics:
        return ""
    return (
        f"{name}: mae={_format_metric(metrics[f'{name}_mae'])} "
        f"rmse={_format_metric(metrics[f'{name}_rmse'])} "
        f"pred={_format_metric(metrics[f'{name}_pred_mean'])} "
        f"target={_format_metric(metrics[f'{name}_target_mean'])}"
    )


def _build_key_val_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    # 统一输出 0-1 区间、越大越好的验证集主指标，便于快速比较实验。
    key_metrics: Dict[str, float] = {}

    regression_metric_names = (
        ("reward", "reward_skill"),
        ("cost", "cost_skill"),
        ("value", "value_skill"),
        ("uncertainty", "uncertainty_skill"),
        ("counterfactual", "counterfactual_skill"),
    )
    for raw_name, display_name in regression_metric_names:
        skill_key = f"{raw_name}_skill"
        if skill_key in metrics:
            key_metrics[display_name] = float(max(0.0, min(1.0, metrics[skill_key])))

    if "done_acc" in metrics:
        key_metrics["done_acc"] = float(max(0.0, min(1.0, metrics["done_acc"])))
    if "valid_f1" in metrics:
        key_metrics["valid_f1"] = float(max(0.0, min(1.0, metrics["valid_f1"])))
    if "valid_exact_match" in metrics:
        key_metrics["valid_exact_match"] = float(max(0.0, min(1.0, metrics["valid_exact_match"])))

    aux_skill_values = [
        float(max(0.0, min(1.0, value)))
        for key, value in metrics.items()
        if key.startswith("aux_") and key.endswith("_skill")
    ]
    if aux_skill_values:
        key_metrics["aux_skill_mean"] = sum(aux_skill_values) / len(aux_skill_values)

    if key_metrics:
        key_metrics["overall"] = sum(key_metrics.values()) / len(key_metrics)
    return key_metrics


def _print_key_val_metrics(key_metrics: Dict[str, float]) -> None:
    if not key_metrics:
        return
    order = (
        "overall",
        "reward_skill",
        "cost_skill",
        "value_skill",
        "uncertainty_skill",
        "counterfactual_skill",
        "done_acc",
        "valid_f1",
        "valid_exact_match",
        "aux_skill_mean",
    )
    parts = [f"{name}={_format_metric(key_metrics[name])}" for name in order if name in key_metrics]
    if parts:
        print(f"  val key metrics: {' '.join(parts)}")


def _print_split_report(split: str, metrics: Dict[str, float], aux_names: Sequence[str]) -> None:
    print(f"  {split} losses: {_format_loss_line(metrics)}")

    regression_names = ("reward", "cost", "value", "uncertainty")
    regression_parts = [part for part in (_format_regression_summary(metrics, name) for name in regression_names) if part]
    if regression_parts:
        print(f"  {split} heads: {' | '.join(regression_parts)}")

    done_parts: List[str] = []
    if "done_acc" in metrics:
        done_parts.append(
            f"done: acc={_format_metric(metrics['done_acc'])} "
            f"brier={_format_metric(metrics['done_brier'])} "
            f"prob={_format_metric(metrics['done_prob_mean'])} "
            f"target={_format_metric(metrics['done_target_mean'])}"
        )
    if "valid_f1" in metrics:
        done_parts.append(
            f"valid: f1={_format_metric(metrics['valid_f1'])} "
            f"prec={_format_metric(metrics['valid_precision'])} "
            f"recall={_format_metric(metrics['valid_recall'])} "
            f"label_acc={_format_metric(metrics['valid_label_acc'])} "
            f"exact={_format_metric(metrics['valid_exact_match'])}"
        )
    if done_parts:
        print(f"  {split} cls: {' | '.join(done_parts)}")

    aux_parts = []
    for name in aux_names:
        metric_key = f"aux_{name}_mae"
        if metric_key in metrics:
            short_name = name.replace("_score", "").replace("termination_readiness", "readiness")
            aux_parts.append(f"{short_name}={_format_metric(metrics[metric_key])}")
    if aux_parts:
        print(f"  {split} aux_mae: {' '.join(aux_parts)}")

    counterfactual_summary = _format_regression_summary(metrics, "counterfactual")
    if counterfactual_summary:
        print(f"  {split} q_head: {counterfactual_summary}")


def train_epoch(
    model: WorkflowWorldModel,
    adapter: WorkflowStateAdapter,
    records: Sequence[Dict],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float,
) -> Dict[str, float]:
    model.train()
    loss_sums: Dict[str, float] = {}
    prediction_stats: Dict[str, Dict[str, float]] = {}
    step_count = 0
    episodes = group_records_by_episode(records)
    for episode_batch in batched_episodes(episodes, batch_size=batch_size, shuffle=True):
        positions = [0 for _ in episode_batch]
        hidden_states: List[Optional[torch.Tensor]] = [None for _ in episode_batch]
        micro_step_count = 0
        total_loss: Optional[torch.Tensor] = None
        optimizer.zero_grad(set_to_none=True)
        while True:
            active_indices = [index for index, episode in enumerate(episode_batch) if positions[index] < len(episode)]
            if not active_indices:
                break
            batch_records = [episode_batch[index][positions[index]] for index in active_indices]
            hidden_rows = [hidden_states[index] for index in active_indices]
            batch_hidden_state = None
            if any(state is not None for state in hidden_rows):
                template = next((state for state in hidden_rows if state is not None), None)
                if template is None:
                    raise RuntimeError("Failed to infer hidden-state template for sequential world-model training.")
                batch_hidden_state = torch.stack(
                    [state if state is not None else template.new_zeros(template.shape) for state in hidden_rows],
                    dim=0,
                )
            batch = adapter.build_batch(batch_records, hidden_state=batch_hidden_state, device=device)
            output = model(batch)
            next_batch = adapter.build_batch(
                build_next_state_records(batch_records),
                hidden_state=output.prior_hidden_state.detach(),
                device=device,
            )
            losses = model.compute_losses(batch, next_batch=next_batch, output=output)
            for name, loss in losses.items():
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss '{name}' detected.")
            total_loss = losses["total"] if total_loss is None else total_loss + losses["total"]
            micro_step_count += 1
            step_count += 1
            _accumulate_batch_metrics(model, loss_sums, prediction_stats, losses, output, batch)
            for row, episode_index in enumerate(active_indices):
                hidden_states[episode_index] = output.prior_hidden_state[row]
                positions[episode_index] += 1
        if total_loss is None or micro_step_count <= 0:
            continue
        total_loss = total_loss / float(micro_step_count)
        total_loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return _finalize_epoch_metrics(loss_sums, prediction_stats, step_count)


@torch.no_grad()
def evaluate_epoch(
    model: WorkflowWorldModel,
    adapter: WorkflowStateAdapter,
    records: Sequence[Dict],
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    # 验证阶段不反传，仅统计各项 loss 和可解释指标。
    if not records:
        return {}
    model.eval()
    loss_sums: Dict[str, float] = {}
    prediction_stats: Dict[str, Dict[str, float]] = {}
    step_count = 0
    episodes = group_records_by_episode(records)
    for episode_batch in batched_episodes(episodes, batch_size=batch_size, shuffle=False):
        positions = [0 for _ in episode_batch]
        hidden_states: List[Optional[torch.Tensor]] = [None for _ in episode_batch]
        while True:
            active_indices = [index for index, episode in enumerate(episode_batch) if positions[index] < len(episode)]
            if not active_indices:
                break
            batch_records = [episode_batch[index][positions[index]] for index in active_indices]
            hidden_rows = [hidden_states[index] for index in active_indices]
            batch_hidden_state = None
            if any(state is not None for state in hidden_rows):
                template = next((state for state in hidden_rows if state is not None), None)
                if template is None:
                    raise RuntimeError("Failed to infer hidden-state template for sequential world-model evaluation.")
                batch_hidden_state = torch.stack(
                    [state if state is not None else template.new_zeros(template.shape) for state in hidden_rows],
                    dim=0,
                )
            batch = adapter.build_batch(batch_records, hidden_state=batch_hidden_state, device=device)
            output = model(batch)
            next_batch = adapter.build_batch(
                build_next_state_records(batch_records),
                hidden_state=output.prior_hidden_state,
                device=device,
            )
            losses = model.compute_losses(batch, next_batch=next_batch, output=output)
            for name, loss in losses.items():
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite validation loss '{name}' detected.")
            step_count += 1
            _accumulate_batch_metrics(model, loss_sums, prediction_stats, losses, output, batch)
            for row, episode_index in enumerate(active_indices):
                hidden_states[episode_index] = output.prior_hidden_state[row]
                positions[episode_index] += 1
    return _finalize_epoch_metrics(loss_sums, prediction_stats, step_count)


def save_checkpoint(
    output_dir: str,
    model: WorkflowWorldModel,
    adapter: WorkflowStateAdapter,
    model_config: WorkflowWorldModelConfig,
    epoch: int,
    metrics: Dict[str, object],
) -> str:
    # 保存模型参数、配置和词表，便于后续恢复训练或接入 planner/reranker。
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_world_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.checkpoint_state_dict(),
            "model_config": model_config,
            "role_to_id": adapter.role_to_id,
            "action_to_id": adapter.action_to_id,
            "task_type_to_id": adapter.task_type_to_id,
            "workflow_state_to_id": adapter.workflow_state_to_id,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_files, train_files, val_files, train_records, val_records = resolve_dataset_splits(args)
    records = train_records + val_records
    dataset_report = build_dataset_report(train_records, val_records, train_files, val_files)
    report_paths = write_dataset_report(args.output_dir, dataset_report)
    print_dataset_report_summary(dataset_report, report_paths)
    next_records = build_next_state_records(records)
    roles, actions = collect_vocab(records)
    adapter = WorkflowStateAdapter(
        agent_roles=roles,
        action_names=actions,
        config=build_adapter_config(args),
    )
    adapter.scan_records(records)
    adapter.scan_records(next_records)

    model_config = build_model_config(adapter)
    adapter.config = model_config
    # 先扫描完当前状态和下一状态的词表，再创建模型并冻结词表。
    adapter.freeze_vocab()
    model = WorkflowWorldModel(model_config).to(args.device)
    tracker = setup_tracking(args, model_config, dataset_size=len(records))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_path = ""
    try:
        for epoch in range(1, args.epochs + 1):
            # 这里暂不引入 lr scheduler、mixed precision、early stopping 等训练工程特性，
            # 先保持训练流程简单、稳定、容易调试。
            train_metrics = train_epoch(
                model=model,
                adapter=adapter,
                records=train_records,
                batch_size=args.batch_size,
                optimizer=optimizer,
                device=args.device,
                gradient_clip=args.gradient_clip,
            )
            val_metrics = evaluate_epoch(
                model=model,
                adapter=adapter,
                records=val_records,
                batch_size=args.batch_size,
                device=args.device,
            )
            key_val_metrics = _build_key_val_metrics(val_metrics)
            current_val = val_metrics.get("total", train_metrics.get("total", 0.0))
            print(f"[Epoch {epoch}] train_total={train_metrics.get('total', 0.0):.4f} val_total={current_val:.4f}")
            _print_split_report("train", train_metrics, model_config.aux_names)
            _print_split_report("val", val_metrics, model_config.aux_names)
            _print_key_val_metrics(key_val_metrics)
            if current_val < best_val:
                # 以验证集 total loss 作为最优指标，保存当前最佳 checkpoint。
                best_val = current_val
                best_path = save_checkpoint(
                    output_dir=args.output_dir,
                    model=model,
                    adapter=adapter,
                model_config=model_config,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics, "key_val_metrics": key_val_metrics},
            )
            log_metrics_to_tracker(tracker, epoch, train_metrics, val_metrics, best_val, key_val_metrics=key_val_metrics)
    finally:
        if tracker is not None:
            tracker.finish()

    print(f"Loaded records: {len(records)} from {len(dataset_files)} files")
    if train_files and val_files:
        print(f"Train files: {len(train_files)} | Eval files: {len(val_files)}")
    print(f"Train records: {len(train_records)} | Validation records: {len(val_records)}")
    print(f"Dataset report JSON: {report_paths['json']}")
    print(f"Dataset report Markdown: {report_paths['markdown']}")
    if best_path:
        print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
