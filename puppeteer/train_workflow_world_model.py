# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import sys
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


def build_adapter_config(args: argparse.Namespace) -> WorkflowWorldModelConfig:
    config = WorkflowWorldModelConfig()
    config.use_llm_text_encoder = bool(args.use_llm_text_encoder)
    config.text_encoder_model_path = str(args.text_encoder_model_path)
    config.text_encoder_freeze = not bool(args.text_encoder_trainable)
    config.text_encoder_dtype = str(args.text_encoder_dtype)
    config.task_text_max_length = int(args.task_text_max_length)
    config.evidence_text_max_length = int(args.evidence_text_max_length)
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
            "use_llm_text_encoder": model_config.use_llm_text_encoder,
            "text_encoder_model_path": model_config.text_encoder_model_path,
            "text_encoder_freeze": model_config.text_encoder_freeze,
            "text_encoder_dtype": model_config.text_encoder_dtype,
            "task_text_max_length": model_config.task_text_max_length,
            "evidence_text_max_length": model_config.evidence_text_max_length,
            "aux_names": list(model_config.aux_names),
            "loss_weights": dict(model_config.loss_weights),
        },
    }


def resolve_dataset_splits(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[Dict], List[Dict], List[Dict]]:
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
) -> None:
    if tracker is None:
        return
    payload: Dict[str, float] = {"epoch": float(epoch)}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            payload[f"{split}/{name}"] = float(value)
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
    if targets.counterfactual_credit is not None:
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
            metrics[f"{name}_pred_mean"] = bucket.get("pred_sum", 0.0) / count
            metrics[f"{name}_target_mean"] = bucket.get("target_sum", 0.0) / count
            metrics[f"{name}_mae"] = bucket.get("abs_sum", 0.0) / count
            metrics[f"{name}_rmse"] = math.sqrt(bucket.get("sq_sum", 0.0) / count)
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
    order = ("total", "latent", "kl", "reward", "cost", "done", "value", "uncertainty", "valid", "aux", "counterfactual")
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
    for batch_records in batched(records, batch_size=batch_size, shuffle=True):
        # 每个 batch 同时构造当前状态和下一状态，用于监督 prior 与 next posterior 对齐。
        batch = adapter.build_batch(batch_records, device=device)
        next_batch = adapter.build_batch(build_next_state_records(batch_records), device=device)
        output = model(batch)
        losses = model.compute_losses(batch, next_batch=next_batch, output=output)
        for name, loss in losses.items():
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss '{name}' detected.")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        step_count += 1
        _accumulate_batch_metrics(model, loss_sums, prediction_stats, losses, output, batch)
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
    for batch_records in batched(records, batch_size=batch_size, shuffle=False):
        batch = adapter.build_batch(batch_records, device=device)
        next_batch = adapter.build_batch(build_next_state_records(batch_records), device=device)
        output = model(batch)
        losses = model.compute_losses(batch, next_batch=next_batch, output=output)
        for name, loss in losses.items():
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite validation loss '{name}' detected.")
        step_count += 1
        _accumulate_batch_metrics(model, loss_sums, prediction_stats, losses, output, batch)
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
            current_val = val_metrics.get("total", train_metrics.get("total", 0.0))
            print(f"[Epoch {epoch}] train_total={train_metrics.get('total', 0.0):.4f} val_total={current_val:.4f}")
            _print_split_report("train", train_metrics, model_config.aux_names)
            _print_split_report("val", val_metrics, model_config.aux_names)
            if current_val < best_val:
                # 以验证集 total loss 作为最优指标，保存当前最佳 checkpoint。
                best_val = current_val
                best_path = save_checkpoint(
                    output_dir=args.output_dir,
                    model=model,
                    adapter=adapter,
                model_config=model_config,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            log_metrics_to_tracker(tracker, epoch, train_metrics, val_metrics, best_val)
    finally:
        if tracker is not None:
            tracker.finish()

    print(f"Loaded records: {len(records)} from {len(dataset_files)} files")
    if train_files and val_files:
        print(f"Train files: {len(train_files)} | Eval files: {len(val_files)}")
    print(f"Train records: {len(train_records)} | Validation records: {len(val_records)}")
    if best_path:
        print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
