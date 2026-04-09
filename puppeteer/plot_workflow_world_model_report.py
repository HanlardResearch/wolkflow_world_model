# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback path depends on runtime env
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize workflow world model analysis artifacts.")
    parser.add_argument(
        "--analysis-json",
        type=str,
        default="checkpoint/workflow_world_model/model_analysis.json",
        help="Path to model_analysis.json produced by analyze_workflow_world_model.py",
    )
    parser.add_argument(
        "--dataset-report-json",
        type=str,
        default="checkpoint/workflow_world_model/dataset_report.json",
        help="Path to dataset_report.json produced during training.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="checkpoint/workflow_world_model/model_predictions.jsonl",
        help="Path to model_predictions.jsonl produced by analyze_workflow_world_model.py",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoint/workflow_world_model/model_report_dashboard.png",
        help="Path to the generated PNG report.",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bar_color(value: float, thresholds: Tuple[float, float] = (0.3, 0.7)) -> str:
    if value >= thresholds[1]:
        return "#2a9d8f"
    if value >= thresholds[0]:
        return "#e9c46a"
    return "#e76f51"


def _safe_metric(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _extract_scatter_points(rows: Iterable[Dict[str, Any]], prediction_key: str, target_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        prediction = row.get("prediction", {})
        target = row.get("target", {})
        try:
            xs.append(float(target[target_key]))
            ys.append(float(prediction[prediction_key]))
        except (KeyError, TypeError, ValueError):
            continue
    return xs, ys


def _top_error_rows(rows: Iterable[Dict[str, Any]], error_key: str, limit: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("error", {}).get(error_key, 0.0)),
        reverse=True,
    )
    return ranked[:limit]


def _format_row_label(row: Dict[str, Any], error_key: str) -> str:
    return (
        f"#{row.get('index', '?')} {row.get('action_name', 'unknown')}\n"
        f"{row.get('workflow_state', 'unknown')}"
        f"\nerr={float(row.get('error', {}).get(error_key, 0.0)):.3f}"
    )


def build_dashboard(analysis_payload: Dict[str, Any], dataset_report: Dict[str, Any], prediction_rows: List[Dict[str, Any]], output_path: str) -> None:
    analysis = analysis_payload.get("analysis", {})
    metrics = analysis.get("metrics", {})
    key_metrics = analysis.get("key_metrics", {})
    conflict_diagnostics = dataset_report.get("conflict_diagnostics", {})

    if plt is None:
        build_svg_dashboard(analysis_payload, dataset_report, prediction_rows, output_path)
        return

    plt.style.use("default")
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.patch.set_facecolor("#f8f6f1")
    for ax in axes.flat:
        ax.set_facecolor("#fffdf8")

    ax = axes[0, 0]
    include_counterfactual = "counterfactual_skill" in key_metrics
    metric_items = [
        ("reward", _safe_metric(key_metrics, "reward_skill")),
        ("cost", _safe_metric(key_metrics, "cost_skill")),
        ("value", _safe_metric(key_metrics, "value_skill")),
        ("uncert", _safe_metric(key_metrics, "uncertainty_skill")),
        ("aux", _safe_metric(key_metrics, "aux_skill_mean")),
        ("done", _safe_metric(key_metrics, "done_acc")),
        ("valid", _safe_metric(key_metrics, "valid_f1")),
    ]
    if include_counterfactual:
        metric_items.insert(4, ("cf", _safe_metric(key_metrics, "counterfactual_skill")))
    names = [name for name, _ in metric_items]
    values = [value for _, value in metric_items]
    colors = [_bar_color(value) for value in values]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Key Validation Metrics")
    ax.set_ylabel("0..1, higher is better")
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.03, 1.03), f"{value:.2f}", ha="center", fontsize=10)

    ax = axes[0, 1]
    summary_lines = []
    for line in analysis.get("strengths", []):
        summary_lines.append(f"Strength: {line}")
    for line in analysis.get("weaknesses", []):
        summary_lines.append(f"Weakness: {line}")
    if conflict_diagnostics:
        for label_name, split_reports in conflict_diagnostics.get("labels", {}).items():
            val_report = split_reports.get("val", {})
            summary_lines.append(
                f"Conflict {label_name}/val: records={float(val_report.get('conflicting_record_ratio', 0.0)):.2%}, "
                f"groups={float(val_report.get('conflicting_group_ratio', 0.0)):.2%}"
            )
    ax.axis("off")
    ax.set_title("Summary Findings")
    ax.text(
        0.01,
        0.99,
        "\n".join(summary_lines) if summary_lines else "No summary data available.",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    ax = axes[1, 0]
    target_means = [
        _safe_metric(metrics, "reward_target_mean"),
        _safe_metric(metrics, "cost_target_mean"),
        _safe_metric(metrics, "value_target_mean"),
        _safe_metric(metrics, "uncertainty_target_mean"),
    ]
    pred_means = [
        _safe_metric(metrics, "reward_pred_mean"),
        _safe_metric(metrics, "cost_pred_mean"),
        _safe_metric(metrics, "value_pred_mean"),
        _safe_metric(metrics, "uncertainty_pred_mean"),
    ]
    labels = ["reward", "cost", "value", "uncert"]
    if "counterfactual_pred_mean" in metrics and "counterfactual_target_mean" in metrics:
        target_means.append(_safe_metric(metrics, "counterfactual_target_mean"))
        pred_means.append(_safe_metric(metrics, "counterfactual_pred_mean"))
        labels.append("cf")
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([item - width / 2 for item in x], target_means, width=width, label="target", color="#577590")
    ax.bar([item + width / 2 for item in x], pred_means, width=width, label="prediction", color="#f3722c")
    ax.set_xticks(x, labels)
    ax.set_title("Prediction Mean vs Target Mean")
    ax.legend()

    ax = axes[1, 1]
    reward_x, reward_y = _extract_scatter_points(prediction_rows, "reward", "reward")
    value_x, value_y = _extract_scatter_points(prediction_rows, "value", "value")
    if reward_x:
        ax.scatter(reward_x, reward_y, s=28, alpha=0.7, label="reward", color="#264653")
    if value_x:
        ax.scatter(value_x, value_y, s=28, alpha=0.7, label="value", color="#e76f51", marker="x")
    low = min(reward_x + value_x + [0.0])
    high = max(reward_x + value_x + [1.0])
    ax.plot([low, high], [low, high], linestyle="--", color="#777777", linewidth=1)
    ax.set_title("Prediction vs Target Scatter")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    if reward_x or value_x:
        ax.legend()

    ax = axes[2, 0]
    worst_rows = _top_error_rows(prediction_rows, "value_abs", limit=5)
    if worst_rows:
        magnitudes = [float(row.get("error", {}).get("value_abs", 0.0)) for row in worst_rows]
        labels = [_format_row_label(row, "value_abs") for row in worst_rows]
        ax.barh(range(len(worst_rows)), magnitudes, color="#d62828")
        ax.set_yticks(range(len(worst_rows)), labels)
        ax.invert_yaxis()
        ax.set_title("Worst Value Predictions")
        ax.set_xlabel("absolute error")
    else:
        ax.axis("off")
        ax.set_title("Worst Value Predictions")
        ax.text(0.02, 0.5, "No prediction rows found.", va="center")

    ax = axes[2, 1]
    if conflict_diagnostics:
        labels = []
        values = []
        for label_name, split_reports in conflict_diagnostics.get("labels", {}).items():
            for split_name in ("train", "val"):
                report = split_reports.get(split_name, {})
                labels.append(f"{label_name}-{split_name}")
                values.append(float(report.get("conflicting_record_ratio", 0.0)))
        colors = [_bar_color(1.0 - min(value * 5.0, 1.0), thresholds=(0.3, 0.7)) for value in values]
        ax.bar(labels, values, color=colors)
        ax.set_title("Repeated Signature Label Conflicts")
        ax.set_ylabel("conflicting record ratio")
        ax.tick_params(axis="x", rotation=20)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.005, f"{value:.1%}", ha="center", fontsize=10)
    else:
        ax.axis("off")
        ax.set_title("Repeated Signature Label Conflicts")
        ax.text(0.02, 0.5, "No conflict diagnostics found in dataset_report.json", va="center")

    fig.suptitle("Workflow World Model Report", fontsize=20, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg_dashboard(analysis_payload: Dict[str, Any], dataset_report: Dict[str, Any], prediction_rows: List[Dict[str, Any]], output_path: str) -> None:
    analysis = analysis_payload.get("analysis", {})
    metrics = analysis.get("metrics", {})
    key_metrics = analysis.get("key_metrics", {})
    conflict_diagnostics = dataset_report.get("conflict_diagnostics", {})
    width = 1600
    height = 1200
    left = 80
    bar_area_width = 560
    bar_height = 28
    gap = 18
    top = 120
    rows: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f6f1"/>',
        '<text x="80" y="60" font-size="32" font-family="Arial" font-weight="bold" fill="#1f2933">Workflow World Model Report</text>',
        '<text x="80" y="92" font-size="16" font-family="Arial" fill="#52606d">Auto-generated from model_analysis.json, dataset_report.json and model_predictions.jsonl</text>',
        '<rect x="60" y="110" width="700" height="420" rx="16" fill="#fffdf8" stroke="#d9d4c7"/>',
        '<text x="80" y="150" font-size="24" font-family="Arial" font-weight="bold" fill="#102a43">Key Validation Metrics</text>',
    ]

    metric_items = [
        ("reward_skill", _safe_metric(key_metrics, "reward_skill")),
        ("cost_skill", _safe_metric(key_metrics, "cost_skill")),
        ("value_skill", _safe_metric(key_metrics, "value_skill")),
        ("uncertainty_skill", _safe_metric(key_metrics, "uncertainty_skill")),
        ("aux_skill_mean", _safe_metric(key_metrics, "aux_skill_mean")),
        ("done_acc", _safe_metric(key_metrics, "done_acc")),
        ("valid_f1", _safe_metric(key_metrics, "valid_f1")),
    ]
    if include_counterfactual:
        metric_items.insert(4, ("counterfactual_skill", _safe_metric(key_metrics, "counterfactual_skill")))
    for idx, (name, value) in enumerate(metric_items):
        y = top + idx * (bar_height + gap)
        rows.append(f'<text x="{left}" y="{y + 20}" font-size="16" font-family="Arial" fill="#243b53">{_svg_escape(name)}</text>')
        rows.append(f'<rect x="{left + 180}" y="{y}" width="{bar_area_width}" height="{bar_height}" rx="10" fill="#ebe7dc"/>')
        rows.append(
            f'<rect x="{left + 180}" y="{y}" width="{bar_area_width * max(0.0, min(value, 1.0)):.1f}" '
            f'height="{bar_height}" rx="10" fill="{_bar_color(value)}"/>'
        )
        rows.append(
            f'<text x="{left + 180 + bar_area_width + 16}" y="{y + 20}" font-size="16" font-family="Arial" fill="#243b53">{value:.3f}</text>'
        )

    rows.extend(
        [
            '<rect x="800" y="110" width="740" height="420" rx="16" fill="#fffdf8" stroke="#d9d4c7"/>',
            '<text x="820" y="150" font-size="24" font-family="Arial" font-weight="bold" fill="#102a43">Summary Findings</text>',
        ]
    )
    summary_lines = []
    for line in analysis.get("strengths", []):
        summary_lines.append(("Strength", line, "#2a9d8f"))
    for line in analysis.get("weaknesses", []):
        summary_lines.append(("Weakness", line, "#d62828"))
    for label_name, split_reports in conflict_diagnostics.get("labels", {}).items():
        val_report = split_reports.get("val", {})
        summary_lines.append(
            (
                "Conflict",
                f"{label_name}/val records={float(val_report.get('conflicting_record_ratio', 0.0)):.1%}, "
                f"groups={float(val_report.get('conflicting_group_ratio', 0.0)):.1%}",
                "#7c6f64",
            )
        )
    for idx, (prefix, line, color) in enumerate(summary_lines[:12]):
        y = 190 + idx * 28
        rows.append(
            f'<text x="820" y="{y}" font-size="18" font-family="Arial" fill="{color}">{_svg_escape(prefix + ": " + line)}</text>'
        )

    rows.extend(
        [
            '<rect x="60" y="560" width="700" height="280" rx="16" fill="#fffdf8" stroke="#d9d4c7"/>',
            '<text x="80" y="600" font-size="24" font-family="Arial" font-weight="bold" fill="#102a43">Prediction Mean vs Target Mean</text>',
        ]
    )
    compare_items = [
        ("reward", _safe_metric(metrics, "reward_target_mean"), _safe_metric(metrics, "reward_pred_mean")),
        ("cost", _safe_metric(metrics, "cost_target_mean"), _safe_metric(metrics, "cost_pred_mean")),
        ("value", _safe_metric(metrics, "value_target_mean"), _safe_metric(metrics, "value_pred_mean")),
        ("uncertainty", _safe_metric(metrics, "uncertainty_target_mean"), _safe_metric(metrics, "uncertainty_pred_mean")),
    ]
    if "counterfactual_pred_mean" in metrics and "counterfactual_target_mean" in metrics:
        compare_items.append(("counterfactual", _safe_metric(metrics, "counterfactual_target_mean"), _safe_metric(metrics, "counterfactual_pred_mean")))
    base_x = 120
    for idx, (name, target_value, pred_value) in enumerate(compare_items):
        x = base_x + idx * 120
        scale = 170
        rows.append(f'<text x="{x}" y="810" font-size="14" font-family="Arial" fill="#243b53">{_svg_escape(name)}</text>')
        rows.append(f'<rect x="{x}" y="{790 - scale * target_value:.1f}" width="28" height="{scale * max(target_value, 0.0):.1f}" fill="#577590"/>')
        rows.append(f'<rect x="{x + 34}" y="{790 - scale * pred_value:.1f}" width="28" height="{scale * max(pred_value, 0.0):.1f}" fill="#f3722c"/>')

    rows.extend(
        [
            '<text x="520" y="640" font-size="14" font-family="Arial" fill="#577590">blue=target</text>',
            '<text x="520" y="662" font-size="14" font-family="Arial" fill="#f3722c">orange=prediction</text>',
            '<rect x="800" y="560" width="740" height="280" rx="16" fill="#fffdf8" stroke="#d9d4c7"/>',
            '<text x="820" y="600" font-size="24" font-family="Arial" font-weight="bold" fill="#102a43">Worst Value Errors</text>',
        ]
    )
    for idx, row in enumerate(_top_error_rows(prediction_rows, "value_abs", limit=5)):
        y = 640 + idx * 36
        rows.append(
            f'<text x="820" y="{y}" font-size="16" font-family="Arial" fill="#243b53">'
            f'{_svg_escape(_format_row_label(row, "value_abs"))}</text>'
        )

    rows.extend(
        [
            '<rect x="60" y="870" width="1480" height="270" rx="16" fill="#fffdf8" stroke="#d9d4c7"/>',
            '<text x="80" y="910" font-size="24" font-family="Arial" font-weight="bold" fill="#102a43">Repeated Signature Label Conflicts</text>',
        ]
    )
    conflict_items: List[Tuple[str, float]] = []
    for label_name, split_reports in conflict_diagnostics.get("labels", {}).items():
        for split_name in ("train", "val"):
            report = split_reports.get(split_name, {})
            conflict_items.append((f"{label_name}-{split_name}", float(report.get("conflicting_record_ratio", 0.0))))
    for idx, (name, value) in enumerate(conflict_items):
        x = 100 + idx * 180
        rows.append(f'<text x="{x}" y="1095" font-size="15" font-family="Arial" fill="#243b53">{_svg_escape(name)}</text>')
        rows.append(f'<rect x="{x}" y="{1060 - value * 120:.1f}" width="80" height="{value * 120:.1f}" fill="{_bar_color(1.0 - min(value * 5.0, 1.0))}"/>')
        rows.append(f'<text x="{x}" y="1080" font-size="14" font-family="Arial" fill="#243b53">{value:.1%}</text>')

    rows.append("</svg>")
    if not output_path.lower().endswith(".svg"):
        output_path = os.path.splitext(output_path)[0] + ".svg"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows))


def main() -> None:
    args = parse_args()
    analysis_payload = _load_json(args.analysis_json)
    dataset_report = _load_json(args.dataset_report_json)
    prediction_rows = _load_jsonl(args.predictions_jsonl)
    build_dashboard(analysis_payload, dataset_report, prediction_rows, args.output_path)
    final_path = args.output_path
    if plt is None and not final_path.lower().endswith(".svg"):
        final_path = os.path.splitext(final_path)[0] + ".svg"
    print(f"[Plot] dashboard={final_path}")


if __name__ == "__main__":
    main()
