# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_workflow_world_model import _normalize_cost, _safe_float, load_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze label/signature conflicts in world model dataset JSONL files.")
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing dataset JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory used to save analysis artifacts.")
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="workflow_world_model",
        help="Only analyze JSONL files whose names start with this prefix.",
    )
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=-1)
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


def _signature(record: Dict[str, Any]) -> Dict[str, str]:
    task = record.get("task", {})
    state = record.get("state", {})
    action = record.get("action", {})
    return {
        "task_type": str(task.get("task_type", task.get("type", "unknown"))).strip() or "unknown",
        "workflow_state": str(state.get("workflow_state", state.get("state", "unknown"))).strip() or "unknown",
        "action_name": str(action.get("name", "unknown")).strip() or "unknown",
        "action_kind": str(action.get("kind", "primitive")).strip() or "primitive",
    }


def _signature_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    item = _signature(record)
    return item["task_type"], item["workflow_state"], item["action_name"]


def _bucket_counts(values: Iterable[str], limit: int = 20) -> List[Dict[str, Any]]:
    counter = Counter(str(value) for value in values)
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    count = float(len(values))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
    }


def _extract_labels(record: Dict[str, Any]) -> Dict[str, float]:
    outcome = record.get("outcome", {})
    next_state_targets = record.get("next_state_targets", {})
    returns = record.get("returns", {})
    return {
        "reward": _safe_float(outcome.get("reward", 0.0)),
        "value": _safe_float(returns.get("mc_return", outcome.get("reward", 0.0))),
        "cost": _normalize_cost(outcome.get("cost_delta", 0.0)),
        "uncertainty": _safe_float(next_state_targets.get("conflict_score", 0.0)),
        "counterfactual": _safe_float(record.get("credit_targets", {}).get("leave_one_out_gap", outcome.get("reward", 0.0))),
        "done": 1.0 if bool(outcome.get("done", False)) else 0.0,
    }


def analyze_conflicts(records: Sequence[Dict[str, Any]], source_files: Sequence[str]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for index, record in enumerate(records):
        key = _signature_key(record)
        grouped.setdefault(key, []).append({"index": index, "record": record, "labels": _extract_labels(record)})

    label_names = ("reward", "value", "cost", "uncertainty", "counterfactual", "done")
    label_conflicts: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    signature_count = len(grouped)
    repeated_signature_count = sum(1 for items in grouped.values() if len(items) > 1)

    for label_name in label_names:
        conflicting_groups: List[Dict[str, Any]] = []
        conflicting_records = 0
        repeated_records = 0
        absolute_ranges: List[float] = []
        for (task_type, workflow_state, action_name), items in grouped.items():
            if len(items) <= 1:
                continue
            repeated_records += len(items)
            values = [float(item["labels"][label_name]) for item in items]
            rounded_unique = sorted({round(value, 6) for value in values})
            if len(rounded_unique) <= 1:
                continue
            conflicting_records += len(items)
            value_range = max(values) - min(values)
            absolute_ranges.append(value_range)
            conflicting_groups.append(
                {
                    "task_type": task_type,
                    "workflow_state": workflow_state,
                    "action_name": action_name,
                    "count": len(items),
                    "unique_values": rounded_unique[:12],
                    "min": min(values),
                    "max": max(values),
                    "range": value_range,
                    "record_indices": [int(item["index"]) for item in items[:20]],
                }
            )
        conflicting_groups.sort(key=lambda item: (-float(item["range"]), -int(item["count"]), str(item["action_name"])))
        label_conflicts[label_name] = {
            "signature_count": signature_count,
            "repeated_signature_count": repeated_signature_count,
            "conflicting_signature_count": len(conflicting_groups),
            "conflicting_signature_ratio": len(conflicting_groups) / max(repeated_signature_count, 1),
            "conflicting_record_count": conflicting_records,
            "conflicting_record_ratio": conflicting_records / max(len(records), 1),
            "range_stats": _summarize_numeric(absolute_ranges),
            "examples": conflicting_groups[:30],
        }
        if conflicting_records / max(len(records), 1) > 0.10:
            warnings.append(
                f"{label_name} has high repeated-signature conflict ratio "
                f"({conflicting_records}/{len(records)} = {conflicting_records / max(len(records), 1):.2%})."
            )

    top_signatures = [
        {
            "task_type": key[0],
            "workflow_state": key[1],
            "action_name": key[2],
            "count": len(items),
        }
        for key, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][2], item[0][1]))[:30]
    ]

    action_counter = Counter()
    workflow_counter = Counter()
    task_counter = Counter()
    for record in records:
        sig = _signature(record)
        action_counter[sig["action_name"]] += 1
        workflow_counter[sig["workflow_state"]] += 1
        task_counter[sig["task_type"]] += 1

    return {
        "overview": {
            "file_count": len(source_files),
            "record_count": len(records),
            "signature_count": signature_count,
            "repeated_signature_count": repeated_signature_count,
            "repeated_signature_ratio": repeated_signature_count / max(signature_count, 1),
        },
        "top_actions": _bucket_counts(action_counter.elements()),
        "top_workflow_states": _bucket_counts(workflow_counter.elements()),
        "top_task_types": _bucket_counts(task_counter.elements()),
        "top_signatures": top_signatures,
        "label_conflicts": label_conflicts,
        "warnings": warnings,
        "source_files": list(source_files),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    overview = report["overview"]
    lines = [
        "# World Model Dataset Conflict Report",
        "",
        "## Overview",
        "",
        f"- Files analyzed: {overview['file_count']}",
        f"- Records analyzed: {overview['record_count']}",
        f"- Unique (task_type, workflow_state, action) signatures: {overview['signature_count']}",
        f"- Repeated signatures: {overview['repeated_signature_count']} ({overview['repeated_signature_ratio']:.2%})",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", []) or ["No major repeated-signature conflicts detected."]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Label Conflicts", ""])
    for label_name, label_report in report["label_conflicts"].items():
        range_stats = label_report["range_stats"]
        lines.extend(
            [
                f"### {label_name}",
                "",
                f"- Conflicting signatures: {label_report['conflicting_signature_count']} / {label_report['repeated_signature_count']} ({label_report['conflicting_signature_ratio']:.2%})",
                f"- Conflicting records: {label_report['conflicting_record_count']} / {overview['record_count']} ({label_report['conflicting_record_ratio']:.2%})",
                f"- Conflict range mean/p50/p75/max: {range_stats['mean']:.4f} / {range_stats['p50']:.4f} / {range_stats['p75']:.4f} / {range_stats['max']:.4f}",
                "",
                "- Example conflicting signatures:",
            ]
        )
        examples = label_report.get("examples", [])[:8]
        if not examples:
            lines.append("- None")
        else:
            for example in examples:
                lines.append(
                    f"- {example['action_name']} @ {example['workflow_state']} "
                    f"count={example['count']} range={example['range']:.4f} values={example['unique_values']}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _bar_color(value: float) -> str:
    if value >= 0.3:
        return "#d62828"
    if value >= 0.1:
        return "#f4a261"
    return "#2a9d8f"


def build_svg_dashboard(report: Dict[str, Any], output_path: str) -> str:
    width = 1600
    height = 1100
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f6f1"/>',
        '<text x="70" y="60" font-size="30" font-family="Arial" font-weight="bold" fill="#1f2933">Dataset Conflict Dashboard</text>',
    ]
    label_items = list(report["label_conflicts"].items())
    top = 130
    bar_left = 320
    bar_width = 520
    for idx, (label_name, label_report) in enumerate(label_items):
        y = top + idx * 70
        value = float(label_report["conflicting_record_ratio"])
        rows.append(f'<text x="80" y="{y + 22}" font-size="20" font-family="Arial" fill="#243b53">{label_name}</text>')
        rows.append(f'<rect x="{bar_left}" y="{y}" width="{bar_width}" height="28" rx="10" fill="#ebe7dc"/>')
        rows.append(
            f'<rect x="{bar_left}" y="{y}" width="{bar_width * min(max(value, 0.0), 1.0):.1f}" height="28" rx="10" fill="{_bar_color(value)}"/>'
        )
        rows.append(f'<text x="{bar_left + bar_width + 20}" y="{y + 22}" font-size="18" font-family="Arial" fill="#243b53">{value:.2%}</text>')

    rows.extend(
        [
            '<text x="70" y="600" font-size="26" font-family="Arial" font-weight="bold" fill="#102a43">Top Conflict Examples</text>',
        ]
    )
    y = 640
    reward_examples = report["label_conflicts"].get("reward", {}).get("examples", [])[:6]
    for example in reward_examples:
        line = (
            f"{example['action_name']} @ {example['workflow_state']} "
            f"count={example['count']} range={example['range']:.2f} values={example['unique_values']}"
        )
        rows.append(f'<text x="80" y="{y}" font-size="16" font-family="Arial" fill="#243b53">{_svg_escape(line)}</text>')
        y += 28

    rows.extend(
        [
            '<text x="70" y="860" font-size="26" font-family="Arial" font-weight="bold" fill="#102a43">Warnings</text>',
        ]
    )
    y = 900
    for warning in (report.get("warnings", []) or ["No major warnings."])[:8]:
        rows.append(f'<text x="80" y="{y}" font-size="17" font-family="Arial" fill="#7f1d1d">{_svg_escape(warning)}</text>')
        y += 28
    rows.append("</svg>")
    final_path = output_path if output_path.lower().endswith(".svg") else os.path.splitext(output_path)[0] + ".svg"
    with open(final_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows))
    return final_path


def _svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_chart(report: Dict[str, Any], output_path: str) -> str:
    if plt is None:
        return build_svg_dashboard(report, output_path)

    labels = list(report["label_conflicts"].keys())
    ratios = [float(report["label_conflicts"][name]["conflicting_record_ratio"]) for name in labels]
    colors = [_bar_color(value) for value in ratios]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#f8f6f1")
    for ax in axes:
        ax.set_facecolor("#fffdf8")

    axes[0].bar(labels, ratios, color=colors)
    axes[0].set_ylim(0.0, max(max(ratios, default=0.1) * 1.2, 0.1))
    axes[0].set_title("Repeated-Signature Conflict Ratio by Label")
    axes[0].set_ylabel("conflicting record ratio")
    for idx, value in enumerate(ratios):
        axes[0].text(idx, value + 0.01, f"{value:.1%}", ha="center")

    axes[1].axis("off")
    examples = report["label_conflicts"].get("reward", {}).get("examples", [])[:10]
    lines = ["Reward conflict examples:"]
    for example in examples:
        lines.append(
            f"{example['action_name']} @ {example['workflow_state']} | "
            f"count={example['count']} range={example['range']:.2f} values={example['unique_values']}"
        )
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(warnings[:8])
    axes[1].text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)

    fig.tight_layout()
    final_path = output_path if output_path.lower().endswith(".png") else os.path.splitext(output_path)[0] + ".png"
    fig.savefig(final_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return final_path


def write_report(output_dir: str, report: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "dataset_conflict_report.json")
    md_path = os.path.join(output_dir, "dataset_conflict_report.md")
    chart_path = os.path.join(output_dir, "dataset_conflict_dashboard.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    final_chart_path = build_chart(report, chart_path)
    return {"json": json_path, "markdown": md_path, "chart": final_chart_path}


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_root}")
    files = find_jsonl_files(args.data_root, args.filename_prefix, args.max_files)
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {args.data_root} with prefix '{args.filename_prefix}'.")
    records = load_records(files, args.max_records)
    report = analyze_conflicts(records, files)
    paths = write_report(args.output_dir, report)
    print(f"[ConflictAnalysis] records={report['overview']['record_count']} signatures={report['overview']['signature_count']}")
    for warning in report.get("warnings", [])[:8]:
        print(f"[ConflictWarning] {warning}")
    print(f"[ConflictAnalysis] json={paths['json']}")
    print(f"[ConflictAnalysis] markdown={paths['markdown']}")
    print(f"[ConflictAnalysis] chart={paths['chart']}")


if __name__ == "__main__":
    main()
