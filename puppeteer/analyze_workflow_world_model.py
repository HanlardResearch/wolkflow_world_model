# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference.policy import WorkflowStateAdapter, WorkflowWorldModel, WorkflowWorldModelConfig  # noqa: E402
from train_workflow_world_model import (  # noqa: E402
    _build_key_val_metrics,
    _finalize_epoch_metrics,
    _format_metric,
    _format_regression_summary,
    _safe_float,
    _accumulate_batch_metrics,
    build_next_state_records,
    resolve_dataset_splits,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline analysis for a trained workflow world model checkpoint.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoint/workflow_world_model/best_world_model.pt",
        help="Path to the saved best_world_model.pt checkpoint.",
    )
    parser.add_argument("--data-root", type=str, default="results", help="Root directory to search for JSONL files.")
    parser.add_argument(
        "--train-data-root",
        type=str,
        default="results/world_model_dataset-llm/MMLU-Pro/train-dataset",
        help="Root directory for training JSONL files. When set with --test-data-root, episode splitting is disabled.",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default="results/world_model_dataset-llm/MMLU-Pro/val-dataset",
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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Episode split ratio when explicit val data is not provided.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint/workflow_world_model",
        help="Directory used to write analysis artifacts.",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=10,
        help="How many highest-error validation samples to keep per head.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=-1,
        help="Limit saved per-sample rows in the JSONL artifact; <=0 keeps all validation samples.",
    )
    return parser.parse_args()


def _coerce_model_config(raw_config: object) -> WorkflowWorldModelConfig:
    if isinstance(raw_config, WorkflowWorldModelConfig):
        return raw_config
    if isinstance(raw_config, dict):
        return WorkflowWorldModelConfig(**raw_config)
    raise TypeError(f"Unsupported model_config type in checkpoint: {type(raw_config)!r}")


def _load_checkpoint(checkpoint_path: str, device: str) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a dict.")
    return checkpoint


def _sorted_vocab_keys(mapping: Dict[str, int]) -> List[str]:
    return [key for key, _ in sorted(mapping.items(), key=lambda item: item[1])]


def build_adapter_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[WorkflowStateAdapter, WorkflowWorldModelConfig]:
    model_config = _coerce_model_config(checkpoint["model_config"])
    role_to_id = dict(checkpoint.get("role_to_id", {}))
    action_to_id = dict(checkpoint.get("action_to_id", {}))
    task_type_to_id = dict(checkpoint.get("task_type_to_id", {}))
    workflow_state_to_id = dict(checkpoint.get("workflow_state_to_id", {}))

    adapter = WorkflowStateAdapter(
        agent_roles=_sorted_vocab_keys(role_to_id),
        action_names=_sorted_vocab_keys(action_to_id),
        config=model_config,
    )
    adapter.role_to_id = role_to_id
    adapter.action_to_id = action_to_id
    adapter.task_type_to_id = task_type_to_id
    adapter.workflow_state_to_id = workflow_state_to_id
    adapter.freeze_vocab()
    return adapter, model_config


def load_model_from_checkpoint(checkpoint: Dict[str, Any], model_config: WorkflowWorldModelConfig, device: str) -> WorkflowWorldModel:
    model = WorkflowWorldModel(model_config).to(device)
    state_dict = checkpoint.get("model_state_dict", {})
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"[Warn] Missing checkpoint keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected checkpoint keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    model.eval()
    return model


def _record_identity(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    task = record.get("task", {})
    state = record.get("state", {})
    action = record.get("action", {})
    outcome = record.get("outcome", {})
    return {
        "index": index,
        "task_type": str(task.get("task_type", task.get("type", "unknown"))),
        "workflow_state": str(state.get("workflow_state", state.get("state", "unknown"))),
        "action_name": str(action.get("name", "unknown")),
        "action_kind": str(action.get("kind", "primitive")),
        "done": bool(outcome.get("done", False)),
        "reward": _safe_float(outcome.get("reward", 0.0)),
    }


def _top_valid_actions(probabilities: torch.Tensor, adapter: WorkflowStateAdapter, limit: int = 5) -> List[Dict[str, float]]:
    scores = probabilities.detach().float().cpu().tolist()
    inverse_vocab = sorted(adapter.action_to_id.items(), key=lambda item: item[1])
    ranked: List[Tuple[str, float]] = []
    for action_name, action_id in inverse_vocab:
        if action_id <= 0 or action_id > len(scores):
            continue
        ranked.append((action_name, float(scores[action_id - 1])))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [{"name": name, "prob": score} for name, score in ranked[:limit]]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def _bucket_counts(values: Iterable[str]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return [{"name": name, "count": count} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def analyze_records(
    model: WorkflowWorldModel,
    adapter: WorkflowStateAdapter,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
    device: str,
    top_k_errors: int,
    sample_limit: int,
) -> Dict[str, Any]:
    loss_sums: Dict[str, float] = {}
    prediction_stats: Dict[str, Dict[str, float]] = {}
    sample_rows: List[Dict[str, Any]] = []
    include_counterfactual = float(model.config.loss_weights.get("counterfactual", 0.0)) > 0
    regression_head_names = ["reward", "cost", "value", "uncertainty"] + (["counterfactual"] if include_counterfactual else [])
    regression_errors: Dict[str, List[Dict[str, Any]]] = {name: [] for name in regression_head_names}
    classification_errors: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ("done", "valid")}
    step_count = 0

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = list(records[start : start + batch_size])
            batch = adapter.build_batch(batch_records, device=device)
            next_batch = adapter.build_batch(build_next_state_records(batch_records), device=device)
            output = model(batch)
            losses = model.compute_losses(batch, next_batch=next_batch, output=output)
            _accumulate_batch_metrics(model, loss_sums, prediction_stats, losses, output, batch)
            step_count += 1

            targets = batch.targets
            done_probs = torch.sigmoid(output.done_logits).detach().float().cpu()
            valid_probs = torch.sigmoid(output.valid_action_logits).detach().float().cpu()
            q_values = model.q_value(output).detach().float().cpu() if include_counterfactual else None
            reward_pred = output.reward.detach().float().cpu()
            cost_pred = output.cost.detach().float().cpu()
            value_pred = output.value.detach().float().cpu()
            uncertainty_pred = output.uncertainty.detach().float().cpu()

            for row_index, record in enumerate(batch_records):
                global_index = start + row_index
                target_row = targets
                identity = _record_identity(record, global_index)
                next_state_targets = record.get("next_state_targets", {})
                true_valid_actions = list(next_state_targets.get("valid_action_mask", record.get("state", {}).get("valid_actions", [])))
                valid_target_count = int(target_row.next_valid_mask[row_index].sum().item()) if target_row and target_row.next_valid_mask is not None else 0

                row = {
                    **identity,
                    "prediction": {
                        "reward": float(reward_pred[row_index].item()),
                        "cost": float(cost_pred[row_index].item()),
                        "value": float(value_pred[row_index].item()),
                        "uncertainty": float(uncertainty_pred[row_index].item()),
                        "done_prob": float(done_probs[row_index].item()),
                        "valid_top_actions": _top_valid_actions(valid_probs[row_index], adapter, limit=5),
                    },
                    "target": {
                        "reward": float(target_row.reward[row_index].item()) if target_row and target_row.reward is not None else 0.0,
                        "cost": float(target_row.cost[row_index].item()) if target_row and target_row.cost is not None else 0.0,
                        "value": float(target_row.value[row_index].item()) if target_row and target_row.value is not None else 0.0,
                        "uncertainty": float(target_row.uncertainty[row_index].item()) if target_row and target_row.uncertainty is not None else 0.0,
                        "done": float(target_row.done[row_index].item()) if target_row and target_row.done is not None else 0.0,
                        "valid_action_count": valid_target_count,
                        "valid_actions": true_valid_actions,
                    },
                    "error": {
                        "reward_abs": abs(float(reward_pred[row_index].item()) - float(target_row.reward[row_index].item()))
                        if target_row and target_row.reward is not None
                        else 0.0,
                        "cost_abs": abs(float(cost_pred[row_index].item()) - float(target_row.cost[row_index].item()))
                        if target_row and target_row.cost is not None
                        else 0.0,
                        "value_abs": abs(float(value_pred[row_index].item()) - float(target_row.value[row_index].item()))
                        if target_row and target_row.value is not None
                        else 0.0,
                        "uncertainty_abs": abs(float(uncertainty_pred[row_index].item()) - float(target_row.uncertainty[row_index].item()))
                        if target_row and target_row.uncertainty is not None
                        else 0.0,
                        "done_abs": abs(float(done_probs[row_index].item()) - float(target_row.done[row_index].item()))
                        if target_row and target_row.done is not None
                        else 0.0,
                    },
                }
                if sample_limit <= 0 or len(sample_rows) < sample_limit:
                    sample_rows.append(row)

                if include_counterfactual and q_values is not None:
                    row["prediction"]["counterfactual_q"] = float(q_values[row_index].item())
                    row["target"]["counterfactual"] = float(target_row.counterfactual_credit[row_index].item()) if target_row and target_row.counterfactual_credit is not None else 0.0
                    row["error"]["counterfactual_abs"] = abs(float(q_values[row_index].item()) - float(target_row.counterfactual_credit[row_index].item())) if target_row and target_row.counterfactual_credit is not None else 0.0

                for head_name, error_key in (
                    ("reward", "reward_abs"),
                    ("cost", "cost_abs"),
                    ("value", "value_abs"),
                    ("uncertainty", "uncertainty_abs"),
                ):
                    regression_errors[head_name].append(
                        {
                            "index": global_index,
                            "action_name": identity["action_name"],
                            "workflow_state": identity["workflow_state"],
                            "target": row["target"].get(head_name if head_name != "counterfactual" else "counterfactual", 0.0),
                            "prediction": row["prediction"].get(head_name if head_name != "counterfactual" else "counterfactual_q", 0.0),
                            "abs_error": row["error"][error_key],
                        }
                    )
                if include_counterfactual and q_values is not None:
                    regression_errors["counterfactual"].append(
                        {
                            "index": global_index,
                            "action_name": identity["action_name"],
                            "workflow_state": identity["workflow_state"],
                            "target": row["target"].get("counterfactual", 0.0),
                            "prediction": row["prediction"].get("counterfactual_q", 0.0),
                            "abs_error": row["error"].get("counterfactual_abs", 0.0),
                        }
                    )

                done_target = float(target_row.done[row_index].item()) if target_row and target_row.done is not None else 0.0
                done_prob = float(done_probs[row_index].item())
                classification_errors["done"].append(
                    {
                        "index": global_index,
                        "action_name": identity["action_name"],
                        "workflow_state": identity["workflow_state"],
                        "target": done_target,
                        "prediction": done_prob,
                        "wrong": int((done_prob >= 0.5) != (done_target >= 0.5)),
                        "confidence_gap": abs(done_prob - done_target),
                    }
                )

                valid_target = target_row.next_valid_mask[row_index].detach().float().cpu() if target_row and target_row.next_valid_mask is not None else None
                if valid_target is not None:
                    valid_pred = valid_probs[row_index] >= 0.5
                    valid_true = valid_target >= 0.5
                    mismatch = int(valid_pred.ne(valid_true).sum().item())
                    classification_errors["valid"].append(
                        {
                            "index": global_index,
                            "action_name": identity["action_name"],
                            "workflow_state": identity["workflow_state"],
                            "target_positive_count": int(valid_true.sum().item()),
                            "pred_positive_count": int(valid_pred.sum().item()),
                            "mismatch_count": mismatch,
                            "exact_match": int(mismatch == 0),
                        }
                    )

    metrics = _finalize_epoch_metrics(loss_sums, prediction_stats, step_count)
    key_metrics = _build_key_val_metrics(metrics)

    for head_name, items in regression_errors.items():
        items.sort(key=lambda item: item["abs_error"], reverse=True)
        regression_errors[head_name] = items[: max(top_k_errors, 0)]
    classification_errors["done"].sort(key=lambda item: (item["wrong"], item["confidence_gap"]), reverse=True)
    classification_errors["valid"].sort(key=lambda item: (item["mismatch_count"], 1 - item["exact_match"]), reverse=True)
    classification_errors["done"] = classification_errors["done"][: max(top_k_errors, 0)]
    classification_errors["valid"] = classification_errors["valid"][: max(top_k_errors, 0)]

    strengths, weaknesses = build_summary_findings(metrics, key_metrics)
    return {
        "metrics": metrics,
        "key_metrics": key_metrics,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "sample_rows": sample_rows,
        "top_errors": {
            "regression": regression_errors,
            "classification": classification_errors,
        },
        "dataset_overview": {
            "record_count": len(records),
            "action_distribution": _bucket_counts(row["action_name"] for row in sample_rows),
            "workflow_state_distribution": _bucket_counts(row["workflow_state"] for row in sample_rows),
        },
    }


def build_summary_findings(metrics: Dict[str, float], key_metrics: Dict[str, float]) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    weaknesses: List[str] = []

    overall = key_metrics.get("overall")
    if overall is not None:
        if overall >= 0.75:
            strengths.append(f"overall key metric is strong ({overall:.3f}).")
        elif overall <= 0.4:
            weaknesses.append(f"overall key metric is weak ({overall:.3f}).")

    for name in ("reward", "cost", "value", "uncertainty", "counterfactual"):
        skill = metrics.get(f"{name}_skill")
        if skill is None:
            continue
        if skill >= 0.7:
            strengths.append(f"{name} head generalizes well (skill={skill:.3f}).")
        elif skill <= 0.2:
            weaknesses.append(f"{name} head is close to a constant baseline (skill={skill:.3f}).")

    done_acc = metrics.get("done_acc")
    done_brier = metrics.get("done_brier")
    if done_acc is not None:
        if done_acc >= 0.95 and (done_brier or 0.0) <= 0.05:
            strengths.append(f"done prediction is reliable (acc={done_acc:.3f}, brier={done_brier:.3f}).")
        elif done_acc <= 0.8:
            weaknesses.append(f"done prediction is unstable (acc={done_acc:.3f}).")

    valid_f1 = metrics.get("valid_f1")
    valid_exact = metrics.get("valid_exact_match")
    if valid_f1 is not None:
        if valid_f1 >= 0.95 and (valid_exact or 0.0) >= 0.9:
            strengths.append(f"valid-action prediction is nearly exact (f1={valid_f1:.3f}, exact={valid_exact:.3f}).")
        elif valid_f1 <= 0.7:
            weaknesses.append(f"valid-action prediction is weak (f1={valid_f1:.3f}).")

    if not strengths:
        strengths.append("no head stands out as clearly strong on the current validation split.")
    if not weaknesses:
        weaknesses.append("no obvious failure head was detected from the current validation metrics.")
    return strengths[:8], weaknesses[:8]


def render_markdown(
    checkpoint_path: str,
    analysis: Dict[str, Any],
    prior_metrics: Dict[str, Any],
) -> str:
    metrics = analysis["metrics"]
    key_metrics = analysis["key_metrics"]
    top_errors = analysis["top_errors"]
    lines = [
        "# World Model Analysis",
        "",
        "## Overview",
        "",
        f"- Checkpoint: {checkpoint_path}",
        f"- Validation records analyzed: {analysis['dataset_overview']['record_count']}",
        f"- Checkpoint epoch: {prior_metrics.get('epoch', 'unknown')}",
        f"- Stored best val total: {_format_metric(_safe_float(prior_metrics.get('best_val_total', float('nan')), float('nan')))}",
        "",
        "## Key Metrics",
        "",
    ]
    for name, value in sorted(key_metrics.items()):
        lines.append(f"- {name}: {_format_metric(value)}")

    lines += [
        "",
        "## Head Summary",
        "",
        f"- reward: {_format_regression_summary(metrics, 'reward') or 'n/a'}",
        f"- cost: {_format_regression_summary(metrics, 'cost') or 'n/a'}",
        f"- value: {_format_regression_summary(metrics, 'value') or 'n/a'}",
        f"- uncertainty: {_format_regression_summary(metrics, 'uncertainty') or 'n/a'}",
        f"- done: acc={_format_metric(metrics.get('done_acc', float('nan')))} brier={_format_metric(metrics.get('done_brier', float('nan')))}",
        f"- valid: f1={_format_metric(metrics.get('valid_f1', float('nan')))} exact={_format_metric(metrics.get('valid_exact_match', float('nan')))}",
        "",
        "## Strengths",
        "",
    ]
    counterfactual_summary = _format_regression_summary(metrics, "counterfactual")
    if counterfactual_summary:
        lines.insert(len(lines) - 4, f"- counterfactual: {counterfactual_summary}")
    for item in analysis["strengths"]:
        lines.append(f"- {item}")
    lines += ["", "## Weaknesses", ""]
    for item in analysis["weaknesses"]:
        lines.append(f"- {item}")

    lines += ["", "## Worst Cases", ""]
    for head_name, items in top_errors["regression"].items():
        if not items:
            continue
        lines.append(f"### {head_name}")
        lines.append("")
        for item in items[:5]:
            lines.append(
                f"- idx={item['index']} action={item['action_name']} state={item['workflow_state']} "
                f"target={_format_metric(item['target'])} pred={_format_metric(item['prediction'])} "
                f"abs_error={_format_metric(item['abs_error'])}"
            )
        lines.append("")

    for head_name, items in top_errors["classification"].items():
        if not items:
            continue
        lines.append(f"### {head_name}")
        lines.append("")
        for item in items[:5]:
            if head_name == "done":
                lines.append(
                    f"- idx={item['index']} action={item['action_name']} state={item['workflow_state']} "
                    f"target={_format_metric(item['target'])} pred={_format_metric(item['prediction'])} "
                    f"wrong={item['wrong']} gap={_format_metric(item['confidence_gap'])}"
                )
            else:
                lines.append(
                    f"- idx={item['index']} action={item['action_name']} state={item['workflow_state']} "
                    f"target_count={item['target_positive_count']} pred_count={item['pred_positive_count']} "
                    f"mismatch={item['mismatch_count']} exact={item['exact_match']}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_prior_checkpoint_metrics(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    metrics = checkpoint.get("metrics", {})
    val_metrics = metrics.get("val", {}) if isinstance(metrics, dict) else {}
    return {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val_total": val_metrics.get("total", float("nan")) if isinstance(val_metrics, dict) else float("nan"),
    }


def write_analysis_artifacts(output_dir: str, analysis: Dict[str, Any], checkpoint_path: str, prior_metrics: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "model_analysis.json")
    md_path = os.path.join(output_dir, "model_analysis.md")
    jsonl_path = os.path.join(output_dir, "model_predictions.jsonl")

    payload = {
        "checkpoint_path": checkpoint_path,
        "prior_checkpoint_metrics": prior_metrics,
        "analysis": {key: value for key, value in analysis.items() if key != "sample_rows"},
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(checkpoint_path, analysis, prior_metrics))
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in analysis["sample_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"json": json_path, "markdown": md_path, "predictions": jsonl_path}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = _load_checkpoint(args.checkpoint_path, args.device)
    adapter, model_config = build_adapter_from_checkpoint(checkpoint)
    model = load_model_from_checkpoint(checkpoint, model_config, args.device)

    _, _, _, _, val_records = resolve_dataset_splits(args)
    analysis = analyze_records(
        model=model,
        adapter=adapter,
        records=val_records,
        batch_size=args.batch_size,
        device=args.device,
        top_k_errors=args.top_k_errors,
        sample_limit=args.sample_limit,
    )
    prior_metrics = _extract_prior_checkpoint_metrics(checkpoint)
    paths = write_analysis_artifacts(args.output_dir, analysis, args.checkpoint_path, prior_metrics)

    print(f"[Analysis] checkpoint={args.checkpoint_path}")
    print(f"[Analysis] val_records={analysis['dataset_overview']['record_count']} overall={_format_metric(analysis['key_metrics'].get('overall', float('nan')))}")
    if analysis["key_metrics"]:
        ordered = " ".join(f"{name}={_format_metric(value)}" for name, value in sorted(analysis["key_metrics"].items()))
        print(f"[Analysis] key_metrics {ordered}")
    print("[Analysis] strengths")
    for item in analysis["strengths"]:
        print(f"  - {item}")
    print("[Analysis] weaknesses")
    for item in analysis["weaknesses"]:
        print(f"  - {item}")
    print(f"[Analysis] json={paths['json']}")
    print(f"[Analysis] markdown={paths['markdown']}")
    print(f"[Analysis] predictions={paths['predictions']}")


if __name__ == "__main__":
    main()
