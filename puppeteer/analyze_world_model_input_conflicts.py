import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reward/value conflicts under the current workflow world model input features."
    )
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing workflow_world_model JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory used to save analysis artifacts.")
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="workflow_world_model",
        help="Only analyze JSONL files whose names start with this prefix.",
    )
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--task-dim", type=int, default=8)
    parser.add_argument("--step-dim", type=int, default=8)
    parser.add_argument("--evidence-dim", type=int, default=8)
    parser.add_argument("--budget-dim", type=int, default=4)
    parser.add_argument("--node-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=6)
    return parser.parse_args()


def find_jsonl_files(data_root: str, filename_prefix: str, max_files: int) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(data_root):
        for file_name in files:
            if not file_name.endswith(".jsonl"):
                continue
            if filename_prefix and not file_name.startswith(filename_prefix):
                continue
            matches.append(os.path.join(root, file_name))
    matches.sort()
    if max_files > 0:
        matches = matches[:max_files]
    return matches


def load_records(paths: Sequence[str], max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))
                if max_records > 0 and len(records) >= max_records:
                    return records
    return records


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and number not in (float("inf"), float("-inf")) else default


def _coarse_signature(record: Dict[str, Any]) -> Tuple[str, str, str]:
    task = record.get("task", {})
    state = record.get("state", {})
    action = record.get("action", {})
    return (
        str(task.get("task_type", task.get("type", "unknown"))).strip() or "unknown",
        str(state.get("workflow_state", state.get("state", "unknown"))).strip() or "unknown",
        str(action.get("name", "unknown")).strip() or "unknown",
    )


def _labels(record: Dict[str, Any]) -> Dict[str, float]:
    outcome = record.get("outcome", {})
    returns = record.get("returns", {})
    return {
        "reward": _safe_float(outcome.get("reward", 0.0)),
        "value": _safe_float(returns.get("mc_return", outcome.get("reward", 0.0))),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    count = float(len(values))
    return {
        "count": count,
        "mean": sum(values) / count,
        "p50": _quantile(values, 0.5),
        "p75": _quantile(values, 0.75),
        "max": max(values),
    }


def _normalize_tokens(value: Any, clip: float = 200000.0) -> float:
    raw = max(_safe_float(value, 0.0), 0.0)
    return min(math.log1p(raw) / math.log1p(clip), 1.0)


def _normalize_cost(value: Any, clip: float = 1.0e8) -> float:
    raw = max(_safe_float(value, 0.0), 0.0)
    return min(math.log1p(raw) / math.log1p(clip), 1.0)


def _text_features(text: str, dim: int) -> List[float]:
    value = str(text or "")
    length = len(value)
    if length <= 0:
        return [0.0] * dim
    words = value.split()
    word_count = len(words)
    digit_count = sum(1 for char in value if char.isdigit())
    punct_count = sum(1 for char in value if not char.isalnum() and not char.isspace())
    url_count = value.count("http://") + value.count("https://")
    unique_ratio = len(set(words)) / max(word_count, 1)
    features = [
        length / 4096.0,
        min(word_count, 512) / 512.0,
        digit_count / max(length, 1),
        punct_count / max(length, 1),
        min(url_count, 4) / 4.0,
        unique_ratio,
        1.0 if "?" in value else 0.0,
        1.0 if "\n" in value else 0.0,
    ]
    return (features + [0.0] * dim)[:dim]


def _id(mapping: Dict[str, int], value: Any) -> int:
    key = str(value or "unknown")
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]


def _scan_vocab(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    vocab = {
        "task_type": {},
        "workflow_state": {},
        "role": {},
        "action": {},
    }
    for record in records:
        task = record.get("task", {})
        state = record.get("state", {})
        _id(vocab["task_type"], task.get("task_type", task.get("type", "unknown")))
        _id(vocab["workflow_state"], state.get("workflow_state", state.get("state", "unknown")))
        for step in state.get("executed_steps", []):
            _id(vocab["role"], step.get("agent", "unknown"))
            _id(vocab["action"], step.get("action", "unknown"))
        for node in record.get("graph", {}).get("nodes", []):
            _id(vocab["role"], node)
        action = record.get("action", {})
        _id(vocab["action"], action.get("name", "unknown"))
        for valid_action in record.get("next_state_targets", {}).get("valid_action_mask", state.get("valid_actions", [])):
            _id(vocab["action"], valid_action)
    return vocab


def _task_question_text(task: Dict[str, Any]) -> str:
    return str(task.get("question", task.get("Question", "")))


def _record_feature_payload(record: Dict[str, Any], vocab: Dict[str, Dict[str, int]], args: argparse.Namespace) -> Dict[str, Any]:
    task = record.get("task", {})
    state = record.get("state", {})
    graph = record.get("graph", {})
    action = record.get("action", {})
    steps = list(state.get("executed_steps", []))[-max(int(args.max_steps), 1) :]

    task_features = _text_features(_task_question_text(task), int(args.task_dim))
    step_entries: List[Dict[str, Any]] = []
    for step in steps:
        step_entries.append(
            {
                "role_id": _id(vocab["role"], step.get("agent", "unknown")),
                "action_id": _id(vocab["action"], step.get("action", "unknown")),
                "features": [
                    1.0 if bool(step.get("success", False)) else 0.0,
                    _normalize_tokens(step.get("tokens", 0.0), clip=200000.0),
                    _normalize_cost(step.get("cost", 0.0), clip=1.0e8),
                    min(len(str(step.get("parameter", ""))), 512) / 512.0,
                    min(len(str(step.get("answer_summary", ""))), 512) / 512.0,
                    min(len(str(step.get("step_data_summary", ""))), 1024) / 1024.0,
                    1.0 if step.get("answer_summary") else 0.0,
                    1.0 if step.get("step_data_summary") else 0.0,
                ][: int(args.step_dim)],
            }
        )

    evidence_kind = {"reasoning": 1, "tool": 2, "answer": 3}
    evidence_items: List[Tuple[str, str]] = []
    evidence_items += [("reasoning", str(item)) for item in state.get("reasoning_results", [])]
    evidence_items += [("tool", str(item)) for item in state.get("tool_results", [])]
    evidence_items += [("answer", str(item)) for item in state.get("recent_answers", [])]
    evidence_entries = [
        {
            "type_id": evidence_kind[kind],
            "features": _text_features(text, int(args.evidence_dim)),
        }
        for kind, text in evidence_items[-max(int(args.max_evidence), 1) :]
    ]

    budget = state.get("budget", {})
    constraints = task.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    budget_features = [
        min(_safe_float(budget.get("step_index", len(steps))), 32.0) / 32.0,
        _normalize_tokens(budget.get("used_tokens", record.get("total_tokens", 0.0)), clip=500000.0),
        _normalize_cost(budget.get("used_cost", record.get("total_cost", 0.0)), clip=1.0e8),
        min(_safe_float(constraints.get("budget", 1.0), 1.0), 1.0),
    ][: int(args.budget_dim)]

    nodes = list(graph.get("nodes", []))[: max(int(args.max_nodes), 1)]
    node_stats = graph.get("node_stats", {})
    node_entries: List[Dict[str, Any]] = []
    for name in nodes:
        stats = node_stats.get(name, {})
        node_entries.append(
            {
                "node_id": _id(vocab["role"], name),
                "features": [
                    _safe_float(stats.get("success_rate", 0.0)),
                    _normalize_cost(stats.get("avg_cost", 0.0), clip=1.0e8),
                    _safe_float(stats.get("avg_credit", 0.0)),
                    min(_safe_float(stats.get("usage_count", 0.0)), 100.0) / 100.0,
                    1.0 if name in {"TavilyAgent", "WebsiteAgent", "ArxivAgent"} else 0.0,
                    1.0 if name == "TerminatorAgent" else 0.0,
                ][: int(args.node_dim)],
            }
        )
    node_slot = {name: index for index, name in enumerate(nodes)}
    edges = []
    for src, dst in graph.get("edges", []):
        if src in node_slot and dst in node_slot:
            edges.append((node_slot[src], node_slot[dst]))

    action_name = str(action.get("name", ""))
    action_features = [
        1.0 if action.get("kind", "primitive") == "macro" else 0.0,
        1.0 if action.get("kind", "primitive") == "mutation" else 0.0,
        1.0 if "Terminator" in action_name else 0.0,
        1.0 if action_name in {"TavilyAgent", "WebsiteAgent", "ArxivAgent"} else 0.0,
        min(len(action_name), 64) / 64.0,
        _normalize_cost(action.get("estimated_cost", 0.0), clip=1.0e8),
    ][: int(args.action_dim)]

    return {
        "task_type_id": _id(vocab["task_type"], task.get("task_type", task.get("type", "unknown"))),
        "workflow_state_id": _id(vocab["workflow_state"], state.get("workflow_state", state.get("state", "unknown"))),
        "task_features": task_features,
        "steps": step_entries,
        "evidence": evidence_entries,
        "budget_features": budget_features,
        "graph_nodes": node_entries,
        "graph_edges": edges,
        "action_kind": str(action.get("kind", "primitive")),
        "action_name_id": _id(vocab["action"], action.get("name", "unknown")),
        "action_features": action_features,
    }


def _build_input_hashes(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    vocab = _scan_vocab(records)
    hashes: List[str] = []
    for record in records:
        payload = _record_feature_payload(record, vocab, args)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        hashes.append(hashlib.sha256(encoded).hexdigest())
    return hashes, vocab


def _group_summary(
    records: Sequence[Dict[str, Any]],
    group_keys: Sequence[Any],
    key_name: str,
) -> Dict[str, Any]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for index, (record, key) in enumerate(zip(records, group_keys)):
        grouped.setdefault(key, []).append({"index": index, "record": record, "labels": _labels(record)})

    repeated_groups = [items for items in grouped.values() if len(items) > 1]
    record_count = len(records)
    repeated_record_count = sum(len(items) for items in repeated_groups)

    labels_report: Dict[str, Any] = {}
    warnings: List[str] = []
    for label_name in ("reward", "value"):
        conflicting_groups: List[Dict[str, Any]] = []
        conflicting_record_count = 0
        ranges: List[float] = []
        for key, items in grouped.items():
            if len(items) <= 1:
                continue
            values = [float(item["labels"][label_name]) for item in items]
            unique_values = sorted({round(value, 6) for value in values})
            if len(unique_values) <= 1:
                continue
            value_range = max(values) - min(values)
            conflicting_record_count += len(items)
            ranges.append(value_range)
            sample = items[0]["record"]
            task = sample.get("task", {})
            state = sample.get("state", {})
            action = sample.get("action", {})
            conflicting_groups.append(
                {
                    "key": str(key),
                    "count": len(items),
                    "range": value_range,
                    "unique_values": unique_values[:12],
                    "task_type": str(task.get("task_type", task.get("type", "unknown"))),
                    "task_id": task.get("id"),
                    "workflow_state": str(state.get("workflow_state", state.get("state", "unknown"))),
                    "action_name": str(action.get("name", "unknown")),
                    "indices": [int(item["index"]) for item in items[:20]],
                }
            )
        conflicting_groups.sort(key=lambda item: (-float(item["range"]), -int(item["count"]), str(item["action_name"])))
        conflicting_ratio = conflicting_record_count / max(record_count, 1)
        if conflicting_ratio > 0.05:
            warnings.append(f"{key_name}/{label_name} conflict ratio is {conflicting_record_count}/{record_count} = {conflicting_ratio:.2%}.")
        labels_report[label_name] = {
            "repeated_group_count": len(repeated_groups),
            "repeated_record_count": repeated_record_count,
            "conflicting_group_count": len(conflicting_groups),
            "conflicting_group_ratio": len(conflicting_groups) / max(len(repeated_groups), 1),
            "conflicting_record_count": conflicting_record_count,
            "conflicting_record_ratio": conflicting_ratio,
            "range_stats": _numeric_summary(ranges),
            "examples": conflicting_groups[:30],
        }

    top_groups: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))[:30]:
        sample = items[0]["record"]
        task = sample.get("task", {})
        state = sample.get("state", {})
        action = sample.get("action", {})
        label_snapshot = _labels(sample)
        top_groups.append(
            {
                "key": str(key),
                "count": len(items),
                "task_type": str(task.get("task_type", task.get("type", "unknown"))),
                "task_id": task.get("id"),
                "workflow_state": str(state.get("workflow_state", state.get("state", "unknown"))),
                "action_name": str(action.get("name", "unknown")),
                "reward_example": label_snapshot["reward"],
                "value_example": label_snapshot["value"],
            }
        )

    return {
        "key_name": key_name,
        "group_count": len(grouped),
        "repeated_group_count": len(repeated_groups),
        "repeated_group_ratio": len(repeated_groups) / max(len(grouped), 1),
        "repeated_record_count": repeated_record_count,
        "repeated_record_ratio": repeated_record_count / max(record_count, 1),
        "labels": labels_report,
        "top_groups": top_groups,
        "warnings": warnings,
    }


def analyze_records(records: Sequence[Dict[str, Any]], source_files: Sequence[str], args: argparse.Namespace) -> Dict[str, Any]:
    coarse_keys = [_coarse_signature(record) for record in records]
    input_hashes, vocab = _build_input_hashes(records, args)

    coarse_summary = _group_summary(records, coarse_keys, "coarse_signature")
    input_summary = _group_summary(records, input_hashes, "model_input_hash")

    comparison: Dict[str, Dict[str, float]] = {}
    for label_name in ("reward", "value"):
        coarse_ratio = float(coarse_summary["labels"][label_name]["conflicting_record_ratio"])
        input_ratio = float(input_summary["labels"][label_name]["conflicting_record_ratio"])
        reduction = 0.0 if coarse_ratio <= 0.0 else max((coarse_ratio - input_ratio) / coarse_ratio, 0.0)
        comparison[label_name] = {
            "coarse_conflicting_record_ratio": coarse_ratio,
            "input_conflicting_record_ratio": input_ratio,
            "relative_reduction": reduction,
        }

    return {
        "config": {
            "feature_mode": "python_reimplementation_of_current_build_batch",
            "max_steps": int(args.max_steps),
            "max_evidence": int(args.max_evidence),
            "max_nodes": int(args.max_nodes),
            "task_dim": int(args.task_dim),
            "step_dim": int(args.step_dim),
            "evidence_dim": int(args.evidence_dim),
            "budget_dim": int(args.budget_dim),
            "node_dim": int(args.node_dim),
            "action_dim": int(args.action_dim),
            "vocab_sizes": {name: len(mapping) for name, mapping in vocab.items()},
        },
        "overview": {
            "file_count": len(source_files),
            "record_count": len(records),
            "unique_input_hash_count": len(set(input_hashes)),
            "unique_coarse_signature_count": len(set(coarse_keys)),
        },
        "coarse_signature": coarse_summary,
        "model_input_hash": input_summary,
        "comparison": comparison,
        "warnings": list(coarse_summary["warnings"]) + list(input_summary["warnings"]),
        "source_files": list(source_files),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    overview = report["overview"]
    comparison = report["comparison"]
    lines = [
        "# World Model Input Conflict Report",
        "",
        "## Overview",
        "",
        f"- Files analyzed: {overview['file_count']}",
        f"- Records analyzed: {overview['record_count']}",
        f"- Unique coarse signatures: {overview['unique_coarse_signature_count']}",
        f"- Unique model input hashes: {overview['unique_input_hash_count']}",
        "",
        "## Key Finding",
        "",
        "This report compares two notions of `same x`:",
        "",
        "- `coarse_signature`: `(task_type, workflow_state, action_name)`",
        "- `model_input_hash`: the actual current model-visible batch tensors hashed row by row",
        "",
    ]
    for label_name in ("reward", "value"):
        stats = comparison[label_name]
        lines.append(
            f"- {label_name}: coarse conflict={stats['coarse_conflicting_record_ratio']:.2%}, "
            f"model-input conflict={stats['input_conflicting_record_ratio']:.2%}, "
            f"relative reduction={stats['relative_reduction']:.2%}"
        )

    warnings = report.get("warnings", [])
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No major conflicts detected.")

    for group_name in ("coarse_signature", "model_input_hash"):
        group_report = report[group_name]
        lines.extend(
            [
                "",
                f"## {group_name}",
                "",
                f"- Repeated groups: {group_report['repeated_group_count']} / {group_report['group_count']} ({group_report['repeated_group_ratio']:.2%})",
                f"- Repeated records: {group_report['repeated_record_count']} / {overview['record_count']} ({group_report['repeated_record_ratio']:.2%})",
                "",
            ]
        )
        for label_name in ("reward", "value"):
            label_report = group_report["labels"][label_name]
            range_stats = label_report["range_stats"]
            lines.extend(
                [
                    f"### {label_name}",
                    "",
                    f"- Conflicting groups: {label_report['conflicting_group_count']} / {label_report['repeated_group_count']} ({label_report['conflicting_group_ratio']:.2%})",
                    f"- Conflicting records: {label_report['conflicting_record_count']} / {overview['record_count']} ({label_report['conflicting_record_ratio']:.2%})",
                    f"- Conflict range mean/p50/p75/max: {range_stats['mean']:.4f} / {range_stats['p50']:.4f} / {range_stats['p75']:.4f} / {range_stats['max']:.4f}",
                    "",
                    "- Example groups:",
                ]
            )
            examples = label_report.get("examples", [])[:8]
            if not examples:
                lines.append("- None")
            else:
                for example in examples:
                    lines.append(
                        f"- action={example['action_name']} task_id={example.get('task_id')} "
                        f"count={example['count']} range={example['range']:.4f} values={example['unique_values']} "
                        f"workflow_state={example['workflow_state']}"
                    )
    return "\n".join(lines).rstrip() + "\n"


def _dashboard_svg(report: Dict[str, Any], output_path: str) -> None:
    width = 1200
    height = 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<text x="40" y="48" font-size="28" font-family="Arial" fill="#102a43">World Model Input Conflict Dashboard</text>',
        '<text x="40" y="76" font-size="14" font-family="Arial" fill="#52606d">Comparing coarse signatures against actual model-visible input hashes</text>',
    ]
    cards = [
        ("reward coarse", report["comparison"]["reward"]["coarse_conflicting_record_ratio"]),
        ("reward model-input", report["comparison"]["reward"]["input_conflicting_record_ratio"]),
        ("value coarse", report["comparison"]["value"]["coarse_conflicting_record_ratio"]),
        ("value model-input", report["comparison"]["value"]["input_conflicting_record_ratio"]),
    ]
    x = 40
    for title, value in cards:
        parts.append(f'<rect x="{x}" y="110" width="250" height="110" rx="14" fill="#ffffff" stroke="#d9e2ec"/>')
        parts.append(f'<text x="{x + 18}" y="146" font-size="18" font-family="Arial" fill="#334e68">{title}</text>')
        parts.append(f'<text x="{x + 18}" y="190" font-size="34" font-family="Arial" fill="#d64545">{value:.2%}</text>')
        x += 280

    base_y = 290
    groups = [("coarse_signature", "#486581"), ("model_input_hash", "#2f855a")]
    for index, (group_name, color) in enumerate(groups):
        gx = 40 + index * 580
        parts.append(f'<rect x="{gx}" y="{base_y}" width="540" height="390" rx="14" fill="#ffffff" stroke="#d9e2ec"/>')
        parts.append(f'<text x="{gx + 18}" y="{base_y + 34}" font-size="22" font-family="Arial" fill="{color}">{group_name}</text>')
        cursor_y = base_y + 72
        for label_name in ("reward", "value"):
            label_report = report[group_name]["labels"][label_name]
            parts.append(
                f'<text x="{gx + 18}" y="{cursor_y}" font-size="16" font-family="Arial" fill="#334e68">'
                f'{label_name}: conflicts={label_report["conflicting_record_count"]}/{report["overview"]["record_count"]} '
                f'({label_report["conflicting_record_ratio"]:.2%})</text>'
            )
            cursor_y += 30
        cursor_y += 12
        parts.append(f'<text x="{gx + 18}" y="{cursor_y}" font-size="16" font-family="Arial" fill="#334e68">Top reward examples:</text>')
        cursor_y += 28
        for example in report[group_name]["labels"]["reward"].get("examples", [])[:5]:
            text = (
                f'{example["action_name"]} | n={example["count"]} | range={example["range"]:.2f} | '
                f'values={",".join(str(v) for v in example["unique_values"][:4])}'
            )
            safe_text = (
                text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            parts.append(
                f'<text x="{gx + 18}" y="{cursor_y}" font-size="13" font-family="Arial" fill="#52606d">{safe_text}</text>'
            )
            cursor_y += 24
    parts.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))


def render_dashboard(report: Dict[str, Any], output_dir: str) -> str:
    if plt is None:
        path = os.path.join(output_dir, "input_conflict_dashboard.svg")
        _dashboard_svg(report, path)
        return path

    labels = ["reward coarse", "reward model-input", "value coarse", "value model-input"]
    values = [
        report["comparison"]["reward"]["coarse_conflicting_record_ratio"],
        report["comparison"]["reward"]["input_conflicting_record_ratio"],
        report["comparison"]["value"]["coarse_conflicting_record_ratio"],
        report["comparison"]["value"]["input_conflicting_record_ratio"],
    ]
    colors = ["#829ab1", "#2f855a", "#829ab1", "#2f855a"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("conflicting record ratio")
    axes[0].set_title("Conflict ratio comparison")
    axes[0].tick_params(axis="x", rotation=20)

    reward_examples = report["model_input_hash"]["labels"]["reward"].get("examples", [])[:8]
    if reward_examples:
        ex_labels = [f'{item["action_name"][:10]}#{index + 1}' for index, item in enumerate(reward_examples)]
        ex_values = [float(item["range"]) for item in reward_examples]
        axes[1].barh(ex_labels, ex_values, color="#d64545")
        axes[1].invert_yaxis()
        axes[1].set_title("Top model-input reward conflict ranges")
        axes[1].set_xlabel("range")
    else:
        axes[1].text(0.5, 0.5, "No model-input reward conflicts", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle("World Model Input Conflict Dashboard")
    fig.tight_layout()
    path = os.path.join(output_dir, "input_conflict_dashboard.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_files = find_jsonl_files(args.data_root, args.filename_prefix, args.max_files)
    if not dataset_files:
        raise FileNotFoundError(f"No JSONL files found under {args.data_root!r} with prefix {args.filename_prefix!r}.")
    records = load_records(dataset_files, args.max_records)
    report = analyze_records(records, dataset_files, args)

    json_path = os.path.join(args.output_dir, "input_conflict_report.json")
    md_path = os.path.join(args.output_dir, "input_conflict_report.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(render_markdown(report))
    dashboard_path = render_dashboard(report, args.output_dir)

    print(f"[InputConflict] json={os.path.relpath(json_path, os.getcwd())}")
    print(f"[InputConflict] markdown={os.path.relpath(md_path, os.getcwd())}")
    print(f"[InputConflict] dashboard={os.path.relpath(dashboard_path, os.getcwd())}")


if __name__ == "__main__":
    main()
