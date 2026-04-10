# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import inspect
import json
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# WorkflowWorldModel 是针对多 agent workflow 的一步世界模型。
# 它把当前任务、执行轨迹、证据、预算、协作图和动作编码成 latent state，
# 再预测下一步 prior latent 以及 reward/cost/done/value/uncertainty/valid action 等头。
# 这个模块主要服务于 imagined rollout、rerank 和 counterfactual credit 估计。


def _default_aux_names() -> Tuple[str, ...]:
    return (
        "progress_score",
        "coverage_score",
        "conflict_score",
        "redundancy_score",
        "termination_readiness",
    )


def _default_loss_weights() -> Dict[str, float]:
    return {
        "latent": 1.0,
        "kl": 0.05,
        "reward": 1.0,
        "cost": 0.5,
        "done": 0.05,
        "value": 0.25,
        "uncertainty": 0.25,
        "valid": 0.05,
        "aux": 0.25,
        "counterfactual": 0.0,
    }


def _default_value_bucket_edges() -> Tuple[float, ...]:
    return (-0.75, -0.25, 0.25, 0.75)


def _default_reward_bucket_edges() -> Tuple[float, ...]:
    return (-0.5, 0.5)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    # 对带 padding 的序列或集合做 masked mean pooling。
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    weighted = values * mask
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return weighted.sum(dim=dim) / denom


def _normalize_adj(adjacency: Tensor, mask: Tensor) -> Tensor:
    # 只保留有效节点之间的边，并按出度做归一化。
    adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _gaussian_kl(post_mean: Tensor, post_logvar: Tensor, prior_mean: Tensor, prior_logvar: Tensor) -> Tensor:
    # 计算 posterior 相对 prior 的 KL，用于约束 latent 的漂移。
    post_var = post_logvar.exp()
    prior_var = prior_logvar.exp()
    return 0.5 * (
        prior_logvar
        - post_logvar
        + (post_var + (post_mean - prior_mean).pow(2)) / prior_var
        - 1.0
    ).sum(dim=-1)


def _pairwise_ranking_loss(prediction: Tensor, target: Tensor) -> Tensor:
    # 反事实 credit 不只关心绝对值，也关心样本间的相对排序。
    if prediction.numel() <= 1:
        return prediction.new_zeros(())
    pred_diff = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
    target_diff = target.unsqueeze(-1) - target.unsqueeze(-2)
    order = torch.sign(target_diff)
    valid = order.ne(0)
    if not valid.any():
        return prediction.new_zeros(())
    return F.softplus(-pred_diff * order.to(dtype=prediction.dtype))[valid].mean()


def _resolve_torch_dtype(dtype_name: str, prefer_cuda: bool) -> torch.dtype:
    name = str(dtype_name or "auto").strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if prefer_cuda and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _extract_last_hidden_state(output: object) -> Tensor:
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, Tensor):
            return first
    raise TypeError("The text encoder output does not contain a last_hidden_state tensor.")


def _coerce_positive_int(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return coerced if coerced > 0 else 0


def _search_hidden_size_in_mapping(mapping: object, candidate_keys: Sequence[str]) -> int:
    if not isinstance(mapping, dict):
        return 0
    for key in candidate_keys:
        size = _coerce_positive_int(mapping.get(key))
        if size > 0:
            return size
    for value in mapping.values():
        if isinstance(value, dict):
            size = _search_hidden_size_in_mapping(value, candidate_keys)
            if size > 0:
                return size
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    size = _search_hidden_size_in_mapping(item, candidate_keys)
                    if size > 0:
                        return size
    return 0


def _infer_hidden_size_from_config(config: object) -> int:
    if config is None:
        return 0
    candidate_keys = (
        "hidden_size",
        "d_model",
        "n_embd",
        "dim",
        "word_embed_proj_dim",
        "embed_dim",
        "model_dim",
    )
    for key in candidate_keys:
        size = _coerce_positive_int(getattr(config, key, None))
        if size > 0:
            return size
    for nested_name in ("text_config", "language_config", "llm_config", "encoder", "decoder", "backbone", "transformer"):
        size = _infer_hidden_size_from_config(getattr(config, nested_name, None))
        if size > 0:
            return size
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            mapping = to_dict()
        except Exception:
            mapping = None
        size = _search_hidden_size_in_mapping(mapping, candidate_keys)
        if size > 0:
            return size
    return 0


def _infer_hidden_size_from_model(model: nn.Module) -> int:
    size = _infer_hidden_size_from_config(getattr(model, "config", None))
    if size > 0:
        return size
    for key in ("hidden_size", "d_model", "n_embd", "dim", "embed_dim", "model_dim"):
        size = _coerce_positive_int(getattr(model, key, None))
        if size > 0:
            return size
    for method_name in ("get_input_embeddings", "get_output_embeddings"):
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        try:
            embedding = method()
        except Exception:
            embedding = None
        if embedding is None:
            continue
        size = _coerce_positive_int(getattr(embedding, "embedding_dim", None))
        if size > 0:
            return size
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, Tensor) and weight.ndim >= 2:
            size = _coerce_positive_int(weight.shape[-1])
            if size > 0:
                return size
    return 0


def _summarize_config_keys(config: object, limit: int = 16) -> str:
    if config is None:
        return "<none>"
    keys: List[str] = []
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            mapping = to_dict()
        except Exception:
            mapping = None
        if isinstance(mapping, dict):
            keys = sorted(str(key) for key in mapping.keys())
    if not keys:
        keys = sorted(key for key in dir(config) if not key.startswith("_"))
    if not keys:
        return "<unavailable>"
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview += ", ..."
    return preview


@dataclass
class WorkflowWorldModelConfig:
    # 各路输入特征维度拆开配置，方便后续替换编码器或扩充特征。
    task_dim: int = 8
    step_numeric_dim: int = 8
    step_text_field_dim: int = 8
    num_step_text_fields: int = 6
    step_dim: int = 16
    evidence_dim: int = 8
    budget_dim: int = 4
    node_dim: int = 6
    action_dim: int = 6
    embed_dim: int = 64
    model_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 256
    dropout: float = 0.1
    num_heads: int = 4
    num_layers: int = 2
    max_steps: int = 16
    max_evidence: int = 12
    max_nodes: int = 24
    num_roles: int = 64
    num_actions: int = 64
    num_task_types: int = 32
    num_workflow_states: int = 32
    use_qwen_text_encoder: bool = False
    qwen_text_encoder_model_name: str = "/extrahome0/HF_models/Qwen/Qwen3-Embedding-0.6B"
    qwen_text_encoder_batch_size: int = 16
    qwen_text_encoder_devices: Tuple[str, ...] = field(default_factory=tuple)
    qwen_text_cache_path: str = ""
    use_llm_text_encoder: bool = False
    text_encoder_model_path: str = ""
    text_encoder_freeze: bool = True
    text_encoder_dtype: str = "auto"
    task_text_max_length: int = 256
    evidence_text_max_length: int = 128
    aux_names: Tuple[str, ...] = field(default_factory=_default_aux_names)
    loss_weights: Dict[str, float] = field(default_factory=_default_loss_weights)
    stable_next_posterior_targets: bool = True
    normalize_kl_by_latent_dim: bool = True
    max_kl_per_sample: float = 5.0
    normalize_latent_alignment: bool = True
    latent_cosine_weight: float = 0.25
    latent_logvar_weight: float = 0.1
    latent_logvar_min: float = -6.0
    latent_logvar_max: float = 2.0
    bound_cost_output: bool = True
    use_reward_buckets: bool = True
    reward_bucket_edges: Tuple[float, ...] = field(default_factory=_default_reward_bucket_edges)
    reward_bucket_label_smoothing: float = 0.02
    use_value_buckets: bool = True
    value_bucket_edges: Tuple[float, ...] = field(default_factory=_default_value_bucket_edges)
    value_bucket_label_smoothing: float = 0.05


@dataclass
class WorkflowWorldModelTargets:
    # 训练阶段可选监督信号，既包含基础环境反馈，也包含更密集的辅助目标。
    reward: Optional[Tensor] = None
    cost: Optional[Tensor] = None
    done: Optional[Tensor] = None
    value: Optional[Tensor] = None
    uncertainty: Optional[Tensor] = None
    next_valid_mask: Optional[Tensor] = None
    aux: Dict[str, Tensor] = field(default_factory=dict)
    counterfactual_credit: Optional[Tensor] = None

    def to(self, device: torch.device | str) -> "WorkflowWorldModelTargets":
        return WorkflowWorldModelTargets(
            reward=None if self.reward is None else self.reward.to(device),
            cost=None if self.cost is None else self.cost.to(device),
            done=None if self.done is None else self.done.to(device),
            value=None if self.value is None else self.value.to(device),
            uncertainty=None if self.uncertainty is None else self.uncertainty.to(device),
            next_valid_mask=None if self.next_valid_mask is None else self.next_valid_mask.to(device),
            aux={name: value.to(device) for name, value in self.aux.items()},
            counterfactual_credit=None
            if self.counterfactual_credit is None
            else self.counterfactual_credit.to(device),
        )


@dataclass
class WorkflowWorldModelBatch:
    # 适配器输出的定长 batch。所有变长字段都已完成 padding。
    task_features: Tensor
    task_type_ids: Tensor
    workflow_state_ids: Tensor
    step_role_ids: Tensor
    step_action_ids: Tensor
    step_features: Tensor
    step_text_field_type_ids: Tensor
    step_text_field_features: Tensor
    step_text_field_mask: Tensor
    step_mask: Tensor
    evidence_type_ids: Tensor
    evidence_features: Tensor
    evidence_mask: Tensor
    budget_features: Tensor
    graph_node_ids: Tensor
    graph_node_features: Tensor
    graph_adj: Tensor
    graph_mask: Tensor
    action_kind_ids: Tensor
    action_name_ids: Tensor
    action_features: Tensor
    task_text_input_ids: Optional[Tensor] = None
    task_text_attention_mask: Optional[Tensor] = None
    evidence_text_input_ids: Optional[Tensor] = None
    evidence_text_attention_mask: Optional[Tensor] = None
    hidden_state: Optional[Tensor] = None
    targets: Optional[WorkflowWorldModelTargets] = None

    @property
    def batch_size(self) -> int:
        return int(self.task_features.shape[0])

    def to(self, device: torch.device | str) -> "WorkflowWorldModelBatch":
        return WorkflowWorldModelBatch(
            task_features=self.task_features.to(device),
            task_type_ids=self.task_type_ids.to(device),
            workflow_state_ids=self.workflow_state_ids.to(device),
            step_role_ids=self.step_role_ids.to(device),
            step_action_ids=self.step_action_ids.to(device),
            step_features=self.step_features.to(device),
            step_text_field_type_ids=self.step_text_field_type_ids.to(device),
            step_text_field_features=self.step_text_field_features.to(device),
            step_text_field_mask=self.step_text_field_mask.to(device),
            step_mask=self.step_mask.to(device),
            evidence_type_ids=self.evidence_type_ids.to(device),
            evidence_features=self.evidence_features.to(device),
            evidence_mask=self.evidence_mask.to(device),
            task_text_input_ids=None if self.task_text_input_ids is None else self.task_text_input_ids.to(device),
            task_text_attention_mask=None
            if self.task_text_attention_mask is None
            else self.task_text_attention_mask.to(device),
            evidence_text_input_ids=None
            if self.evidence_text_input_ids is None
            else self.evidence_text_input_ids.to(device),
            evidence_text_attention_mask=None
            if self.evidence_text_attention_mask is None
            else self.evidence_text_attention_mask.to(device),
            budget_features=self.budget_features.to(device),
            graph_node_ids=self.graph_node_ids.to(device),
            graph_node_features=self.graph_node_features.to(device),
            graph_adj=self.graph_adj.to(device),
            graph_mask=self.graph_mask.to(device),
            action_kind_ids=self.action_kind_ids.to(device),
            action_name_ids=self.action_name_ids.to(device),
            action_features=self.action_features.to(device),
            hidden_state=None if self.hidden_state is None else self.hidden_state.to(device),
            targets=None if self.targets is None else self.targets.to(device),
        )


@dataclass
class WorkflowWorldModelOutput:
    # 模型前向输出，既包含 latent，也包含各个任务头的预测结果。
    latent: Tensor
    latent_mean: Tensor
    latent_logvar: Tensor
    hidden_state: Tensor
    graph_embedding: Tensor
    action_embedding: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor
    prior_hidden_state: Tensor
    reward: Tensor
    reward_logits: Optional[Tensor]
    cost: Tensor
    done_logits: Tensor
    value: Tensor
    value_logits: Optional[Tensor]
    uncertainty: Tensor
    valid_action_logits: Tensor
    aux: Dict[str, Tensor]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class StructuredStateSpaceCell(nn.Module):
    # 轻量 SSM 单元：用连续衰减状态聚合前缀信息。
    def __init__(self, input_dim: int, state_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, state_dim * 2)
        self.output_projection = nn.Linear(state_dim, output_dim)
        self.skip_projection = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_decay = nn.Parameter(torch.zeros(state_dim))

    def forward(self, inputs: Tensor, prev_state: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if prev_state is None:
            prev_state = inputs.new_zeros(inputs.shape[0], self.log_decay.shape[0])
        drive, gate = self.input_projection(inputs).chunk(2, dim=-1)
        decay = torch.sigmoid(self.log_decay).unsqueeze(0)
        next_state = decay * prev_state + (1.0 - decay) * torch.tanh(drive)
        mixed = torch.tanh(next_state) * torch.sigmoid(gate)
        output = self.output_projection(mixed) + self.skip_projection(inputs)
        output = self.output_norm(output)
        output = self.dropout(output)
        return output, next_state


class SequenceEncoder(nn.Module):
    # 对历史 step 序列编码，输入包含角色、动作、数值特征和字段级文本特征。
    def __init__(self, num_roles: int, num_actions: int, numeric_dim: int, config: WorkflowWorldModelConfig) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles + 1, config.embed_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_actions + 1, config.embed_dim, padding_idx=0)
        self.numeric_projection = nn.Linear(numeric_dim, config.model_dim)
        self.text_field_type_embedding = nn.Embedding(
            config.num_step_text_fields + 1,
            config.model_dim,
            padding_idx=0,
        )
        self.step_text_presence_embedding = nn.Embedding(2, config.embed_dim, padding_idx=0)
        self.text_field_projection = nn.Linear(config.step_text_field_dim, config.model_dim)
        self.text_gate = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.input_projection = nn.Linear(config.embed_dim * 3 + config.model_dim * 2, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def _pool_step_text_fields(
        self,
        numeric_repr: Tensor,
        text_field_type_ids: Tensor,
        text_field_features: Tensor,
        text_field_mask: Tensor,
    ) -> Tensor:
        if text_field_features.shape[-1] == 0:
            return numeric_repr.new_zeros(numeric_repr.shape)
        type_repr = self.text_field_type_embedding(text_field_type_ids)
        value_repr = self.text_field_projection(text_field_features)
        field_repr = value_repr + type_repr
        step_context = numeric_repr.unsqueeze(2).expand(-1, -1, field_repr.shape[2], -1)
        gate_logits = self.text_gate(torch.cat([field_repr, step_context], dim=-1)).squeeze(-1)
        valid_mask = text_field_mask > 0
        gate_logits = gate_logits.masked_fill(~valid_mask, -1.0e9)
        attention = torch.softmax(gate_logits, dim=-1)
        attention = attention * valid_mask.to(dtype=attention.dtype)
        denom = attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        attention = attention / denom
        return (field_repr * attention.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        role_ids: Tensor,
        action_ids: Tensor,
        features: Tensor,
        text_field_type_ids: Tensor,
        text_field_features: Tensor,
        text_field_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # 输出整个 step 序列的 pooled 表示，而不是逐 token 表示。
        # 对于没有任何历史 step 的样本，直接返回零向量，避免 Transformer
        # 在“整行都被 mask”时产生 NaN。
        role_repr = self.role_embedding(role_ids)
        action_repr = self.action_embedding(action_ids)
        numeric_repr = self.numeric_projection(features)
        text_repr = self._pool_step_text_fields(numeric_repr, text_field_type_ids, text_field_features, text_field_mask)
        step_has_text = (text_field_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=role_repr.dtype)
        token_repr = self.input_projection(
            torch.cat(
                [
                    role_repr,
                    action_repr,
                    self.step_text_presence_embedding((step_has_text.squeeze(-1)).long()),
                    numeric_repr,
                    text_repr,
                ],
                dim=-1,
            )
        )
        has_tokens = mask.sum(dim=1) > 0
        pooled = token_repr.new_zeros(token_repr.shape[0], token_repr.shape[-1])
        if has_tokens.any():
            valid_tokens = token_repr[has_tokens]
            valid_mask = mask[has_tokens]
            encoded = self.encoder(valid_tokens, src_key_padding_mask=valid_mask.eq(0))
            pooled[has_tokens] = _masked_mean(encoded, valid_mask, dim=1)
        return pooled


class SetEncoder(nn.Module):
    # 对证据集合做无序编码，适合 reasoning/tool/answer 这类 item set。
    def __init__(self, num_types: int, feature_dim: int, config: WorkflowWorldModelConfig) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(num_types + 1, config.embed_dim, padding_idx=0)
        self.feature_projection = nn.Linear(feature_dim, config.model_dim)
        self.output_projection = nn.Linear(config.embed_dim + config.model_dim, config.model_dim)

    def forward(self, type_ids: Tensor, features: Tensor, mask: Tensor, preprojected: bool = False) -> Tensor:
        type_repr = self.type_embedding(type_ids)
        value_repr = features if preprojected else self.feature_projection(features)
        item_repr = self.output_projection(torch.cat([type_repr, value_repr], dim=-1))
        return _masked_mean(item_repr, mask, dim=1)


class GraphEncoder(nn.Module):
    # 编码当前协作图，结合节点属性和邻接关系生成图级表示。
    def __init__(self, config: WorkflowWorldModelConfig) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(config.num_roles + 1, config.embed_dim, padding_idx=0)
        self.node_projection = nn.Linear(config.node_dim, config.model_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embed_dim + config.model_dim * 2, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.model_dim),
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, node_ids: Tensor, node_features: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        # 每层都执行一次“邻居聚合 + 自身更新”，最后再做 masked pooling。
        role_repr = self.role_embedding(node_ids)
        hidden = self.node_projection(node_features)
        norm_adj = _normalize_adj(adjacency, mask)
        for layer in self.layers:
            neighbors = torch.bmm(norm_adj, hidden)
            hidden = layer(torch.cat([role_repr, hidden, neighbors], dim=-1))
        return _masked_mean(hidden, mask, dim=1)


class HFTextEncoder(nn.Module):
    # 使用 HuggingFace 文本模型对问题和证据做语义编码，可选择冻结参数。
    def __init__(self, model_path: str, freeze: bool, dtype_name: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required when --use-llm-text-encoder is enabled.") from exc

        if not model_path:
            raise ValueError("A text encoder model path must be provided when LLM text encoding is enabled.")

        prefer_cuda = torch.cuda.is_available()
        torch_dtype = _resolve_torch_dtype(dtype_name, prefer_cuda=prefer_cuda)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()
        self.hidden_size = _infer_hidden_size_from_model(self.model)
        if self.hidden_size <= 0:
            config_name = type(getattr(self.model, "config", None)).__name__
            config_keys = _summarize_config_keys(getattr(self.model, "config", None))
            raise ValueError(
                "Failed to infer hidden_size from the text encoder config. "
                f"config_type={config_name}, available_keys={config_keys}"
            )
        signature = inspect.signature(self.model.forward)
        self._supports_use_cache = "use_cache" in signature.parameters

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(self.training)
        context = torch.no_grad() if self.freeze else nullcontext()
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if self._supports_use_cache:
            kwargs["use_cache"] = False
        with context:
            outputs = self.model(**kwargs)
        hidden = _extract_last_hidden_state(outputs)
        return _masked_mean(hidden, attention_mask, dim=1)


class QwenTextEmbeddingEncoder:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        devices: Optional[Sequence[str]] = None,
        cache_path: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required when the Qwen text embedding encoder is enabled."
            ) from exc
        if not model_name:
            raise ValueError("qwen_text_encoder_model_name must be set when the Qwen text encoder is enabled.")
        normalized_devices = [str(device).strip() for device in (devices or []) if str(device).strip()]
        primary_device = normalized_devices[0] if normalized_devices else None
        load_device = "cpu" if len(normalized_devices) > 1 else primary_device
        self.model = SentenceTransformer(model_name, device=load_device)
        self.batch_size = max(int(batch_size), 1)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = int(self.model.get_embedding_dimension())
        else:
            self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        self._cache: Dict[Tuple[str, str], List[float]] = {}
        self.devices: Tuple[str, ...] = tuple(normalized_devices)
        self._pool: Optional[Dict[str, Any]] = None
        self.cache_path = str(cache_path or "").strip()
        if self.cache_path:
            self._load_cache(self.cache_path)

    def encode(self, texts: Sequence[str], prompt_name: Optional[str] = None) -> List[List[float]]:
        normalized_texts = [str(text or "") for text in texts]
        results: List[Optional[List[float]]] = [None] * len(normalized_texts)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        cache_key_prefix = str(prompt_name or "")
        for index, text in enumerate(normalized_texts):
            cache_key = (cache_key_prefix, text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[index] = list(cached)
                continue
            uncached_texts.append(text)
            uncached_indices.append(index)
        if uncached_texts:
            kwargs: Dict[str, Any] = {
                "batch_size": self.batch_size,
                "convert_to_tensor": False,
            }
            if prompt_name:
                kwargs["prompt_name"] = prompt_name
            if self._pool is not None and not prompt_name:
                kwargs.pop("prompt_name", None)
                embeddings = self.model.encode_multi_process(
                    uncached_texts,
                    pool=self._pool,
                    batch_size=self.batch_size,
                )
            else:
                if len(self.devices) > 1 and not prompt_name:
                    self._ensure_pool()
                if self._pool is not None and not prompt_name:
                    embeddings = self.model.encode_multi_process(
                        uncached_texts,
                        pool=self._pool,
                        batch_size=self.batch_size,
                    )
                else:
                    embeddings = self.model.encode(uncached_texts, **kwargs)
            for offset, embedding in enumerate(embeddings):
                values = [float(value) for value in embedding.tolist()]
                index = uncached_indices[offset]
                cache_key = (cache_key_prefix, normalized_texts[index])
                self._cache[cache_key] = values
                results[index] = values
        return [result if result is not None else [0.0] * self.embedding_dim for result in results]

    def close(self) -> None:
        if self._pool is not None:
            self.model.stop_multi_process_pool(self._pool)
            self._pool = None

    def _ensure_pool(self) -> None:
        if self._pool is None and len(self.devices) > 1:
            self._pool = self.model.start_multi_process_pool(list(self.devices))
            atexit.register(self.close)

    def _load_cache(self, cache_path: str) -> None:
        if not cache_path:
            return
        paths: List[str] = []
        if os.path.isdir(cache_path):
            for name in sorted(os.listdir(cache_path)):
                if name.endswith(".jsonl"):
                    paths.append(os.path.join(cache_path, name))
        elif os.path.isfile(cache_path):
            paths.append(cache_path)
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    prompt_name = str(payload.get("prompt_name", ""))
                    text = str(payload.get("text", ""))
                    embedding = payload.get("embedding", [])
                    if not isinstance(embedding, list):
                        continue
                    self._cache[(prompt_name, text)] = [float(value) for value in embedding]


class WorkflowWorldModel(nn.Module):
    # 三个核心阶段：
    # 1. 从当前观测编码 posterior latent。
    # 2. 结合 action 和 graph 做 transition，得到下一步 prior latent。
    # 3. 基于 prior 预测 reward/cost/done/value/uncertainty/valid action 等头。
    def __init__(self, config: WorkflowWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder: Optional[HFTextEncoder] = None
        self.task_type_embedding = nn.Embedding(config.num_task_types + 1, config.embed_dim, padding_idx=0)
        self.workflow_state_embedding = nn.Embedding(config.num_workflow_states + 1, config.embed_dim, padding_idx=0)
        self.action_kind_embedding = nn.Embedding(4, config.embed_dim, padding_idx=0)
        self.action_name_embedding = nn.Embedding(config.num_actions + 1, config.embed_dim, padding_idx=0)
        self.task_projection = MLP(config.task_dim, config.hidden_dim, config.model_dim, config.dropout)
        self.budget_projection = MLP(config.budget_dim, config.hidden_dim // 2, config.model_dim, config.dropout)
        self.step_encoder = SequenceEncoder(config.num_roles, config.num_actions, config.step_numeric_dim, config)
        self.evidence_encoder = SetEncoder(3, config.evidence_dim, config)
        self.graph_encoder = GraphEncoder(config)
        self.action_feature_projection = nn.Linear(config.action_dim, config.model_dim)
        self.action_projection = nn.Linear(config.embed_dim * 2 + config.model_dim, config.model_dim)
        if config.use_llm_text_encoder:
            self.text_encoder = HFTextEncoder(
                model_path=config.text_encoder_model_path,
                freeze=config.text_encoder_freeze,
                dtype_name=config.text_encoder_dtype,
            )
            self.task_text_projection = nn.Linear(self.text_encoder.hidden_size, config.model_dim)
            self.evidence_text_projection = nn.Linear(self.text_encoder.hidden_size, config.model_dim)
        else:
            self.task_text_projection = None
            self.evidence_text_projection = None
        self.observation_fusion = nn.Sequential(
            nn.Linear(config.model_dim * 5 + config.embed_dim * 2 + config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.observation_state_cell = StructuredStateSpaceCell(
            input_dim=config.hidden_dim,
            state_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.posterior_head = nn.Linear(config.hidden_dim, config.latent_dim * 2)
        self.transition_input = nn.Linear(config.latent_dim + config.model_dim * 2, config.hidden_dim)
        self.transition_cell = StructuredStateSpaceCell(
            input_dim=config.hidden_dim,
            state_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.prior_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )
        head_dim = config.latent_dim + config.model_dim + config.hidden_dim
        if config.use_reward_buckets:
            self.reward_head = MLP(head_dim, config.hidden_dim, len(config.reward_bucket_edges) + 1, config.dropout)
        else:
            self.reward_head = MLP(head_dim, config.hidden_dim, 1, config.dropout)
        self.cost_head = MLP(head_dim, config.hidden_dim, 1, config.dropout)
        self.done_head = MLP(head_dim, config.hidden_dim, 1, config.dropout)
        if config.use_value_buckets:
            self.value_head = MLP(head_dim, config.hidden_dim, len(config.value_bucket_edges) + 1, config.dropout)
        else:
            self.value_head = MLP(head_dim, config.hidden_dim, 1, config.dropout)
        self.uncertainty_head = MLP(head_dim, config.hidden_dim, 1, config.dropout)
        self.valid_head = MLP(config.latent_dim + config.model_dim, config.hidden_dim, config.num_actions, config.dropout)
        self.aux_heads = nn.ModuleDict(
            {name: MLP(head_dim, config.hidden_dim, 1, config.dropout) for name in config.aux_names}
        )
        bucket_centers = self._build_value_bucket_centers(config.value_bucket_edges)
        self.register_buffer("value_bucket_centers", torch.tensor(bucket_centers, dtype=torch.float32), persistent=False)
        reward_bucket_centers = self._build_value_bucket_centers(config.reward_bucket_edges)
        self.register_buffer("reward_bucket_centers", torch.tensor(reward_bucket_centers, dtype=torch.float32), persistent=False)

    @classmethod
    def from_adapter(
        cls,
        adapter: "WorkflowStateAdapter",
        config: Optional[WorkflowWorldModelConfig] = None,
    ) -> "WorkflowWorldModel":
        # 根据 adapter 扫描到的词表规模修正 embedding 大小，再创建模型。
        config = replace(adapter.config) if config is None else config
        config.num_roles = max(len(adapter.role_to_id), 1)
        config.num_actions = max(len(adapter.action_to_id), 1)
        config.num_task_types = max(len(adapter.task_type_to_id), 1)
        config.num_workflow_states = max(len(adapter.workflow_state_to_id), 1)
        return cls(config)

    def _sample_latent(self, mean: Tensor, logvar: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    @contextmanager
    def _temporary_eval_mode(self, enabled: bool):
        if not enabled:
            yield
            return
        was_training = self.training
        if was_training:
            self.eval()
        try:
            yield
        finally:
            if was_training:
                self.train(was_training)

    def _encode_transition_target(self, batch: WorkflowWorldModelBatch) -> Tuple[Tensor, Tensor]:
        # Use deterministic posterior targets so the prior does not chase dropout noise.
        with torch.no_grad():
            with self._temporary_eval_mode(self.config.stable_next_posterior_targets):
                _, next_mean, next_logvar, _, _ = self.encode_observation(batch, sample_latent=False)
        return next_mean.detach(), next_logvar.detach()

    def _reduce_kl_loss(self, kl_values: Tensor) -> Tensor:
        if self.config.normalize_kl_by_latent_dim and self.config.latent_dim > 0:
            kl_values = kl_values / float(self.config.latent_dim)
        max_kl = float(self.config.max_kl_per_sample)
        if max_kl > 0:
            kl_values = kl_values.clamp(max=max_kl)
        return kl_values.mean()

    def _normalize_latent_for_alignment(self, latent: Tensor) -> Tensor:
        if not self.config.normalize_latent_alignment:
            return latent
        return F.layer_norm(latent, (latent.shape[-1],))

    def _latent_alignment_loss(self, prior_mean: Tensor, target_mean: Tensor) -> Tensor:
        prior_view = self._normalize_latent_for_alignment(prior_mean)
        target_view = self._normalize_latent_for_alignment(target_mean)
        loss = F.smooth_l1_loss(prior_view, target_view)
        cosine_weight = float(self.config.latent_cosine_weight)
        if cosine_weight > 0:
            cosine_loss = 1.0 - F.cosine_similarity(prior_view, target_view, dim=-1, eps=1.0e-8).mean()
            loss = loss + cosine_weight * cosine_loss
        return loss

    def _cost_output(self, raw_cost: Tensor) -> Tensor:
        if self.config.bound_cost_output:
            return torch.sigmoid(raw_cost)
        return raw_cost

    @staticmethod
    def _build_value_bucket_centers(edges: Sequence[float]) -> List[float]:
        ordered = sorted(float(edge) for edge in edges)
        boundaries = [-1.0] + ordered + [1.0]
        centers: List[float] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            centers.append((left + right) * 0.5)
        return centers

    def _decode_value_output(self, raw_value: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.config.use_value_buckets:
            return raw_value.squeeze(-1), None
        value_logits = raw_value
        probabilities = torch.softmax(value_logits, dim=-1)
        centers = self.value_bucket_centers.to(device=value_logits.device, dtype=value_logits.dtype)
        value = (probabilities * centers.unsqueeze(0)).sum(dim=-1)
        return value, value_logits

    def _decode_reward_output(self, raw_reward: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.config.use_reward_buckets:
            return raw_reward.squeeze(-1), None
        reward_logits = raw_reward
        probabilities = torch.softmax(reward_logits, dim=-1)
        centers = self.reward_bucket_centers.to(device=reward_logits.device, dtype=reward_logits.dtype)
        reward = (probabilities * centers.unsqueeze(0)).sum(dim=-1)
        return reward, reward_logits

    def _value_bucket_targets(self, target_value: Tensor) -> Tensor:
        edges = torch.tensor(
            list(self.config.value_bucket_edges),
            device=target_value.device,
            dtype=target_value.dtype,
        )
        clipped = target_value.clamp(min=-1.0, max=1.0)
        return torch.bucketize(clipped, edges)

    def _reward_bucket_targets(self, target_reward: Tensor) -> Tensor:
        edges = torch.tensor(
            list(self.config.reward_bucket_edges),
            device=target_reward.device,
            dtype=target_reward.dtype,
        )
        clipped = target_reward.clamp(min=-1.0, max=1.0)
        return torch.bucketize(clipped, edges)

    def encode_action(self, kind_ids: Tensor, name_ids: Tensor, features: Tensor) -> Tensor:
        # 动作编码包含动作类型、动作名和少量数值特征。
        kind_repr = self.action_kind_embedding(kind_ids)
        name_repr = self.action_name_embedding(name_ids)
        numeric_repr = self.action_feature_projection(features)
        return self.action_projection(torch.cat([kind_repr, name_repr, numeric_repr], dim=-1))

    def _encode_text_inputs(self, input_ids: Tensor, attention_mask: Tensor, projection: nn.Linear) -> Tensor:
        if self.text_encoder is None:
            raise RuntimeError("LLM text encoder is not initialized.")
        pooled = self.text_encoder(input_ids, attention_mask)
        pooled = pooled.to(device=projection.weight.device, dtype=projection.weight.dtype)
        return projection(pooled)

    def _encode_task_inputs(self, batch: WorkflowWorldModelBatch) -> Tensor:
        if not self.config.use_llm_text_encoder:
            return self.task_projection(batch.task_features)
        if batch.task_text_input_ids is None or batch.task_text_attention_mask is None or self.task_text_projection is None:
            raise ValueError("Task text tokens are missing from the batch.")
        return self._encode_text_inputs(
            batch.task_text_input_ids,
            batch.task_text_attention_mask,
            self.task_text_projection,
        )

    def _encode_evidence_inputs(self, batch: WorkflowWorldModelBatch) -> Tensor:
        if not self.config.use_llm_text_encoder:
            return self.evidence_encoder(
                batch.evidence_type_ids,
                batch.evidence_features,
                batch.evidence_mask,
            )
        if (
            batch.evidence_text_input_ids is None
            or batch.evidence_text_attention_mask is None
            or self.evidence_text_projection is None
        ):
            raise ValueError("Evidence text tokens are missing from the batch.")
        batch_size, num_items, seq_len = batch.evidence_text_input_ids.shape
        flat_ids = batch.evidence_text_input_ids.reshape(batch_size * num_items, seq_len)
        flat_attention_mask = batch.evidence_text_attention_mask.reshape(batch_size * num_items, seq_len)
        flat_valid_mask = batch.evidence_mask.reshape(batch_size * num_items) > 0
        flat_repr = batch.evidence_features.new_zeros(batch_size * num_items, self.config.model_dim)
        if flat_valid_mask.any():
            flat_repr[flat_valid_mask] = self._encode_text_inputs(
                flat_ids[flat_valid_mask],
                flat_attention_mask[flat_valid_mask],
                self.evidence_text_projection,
            ).to(flat_repr.dtype)
        evidence_repr = flat_repr.reshape(batch_size, num_items, self.config.model_dim)
        return self.evidence_encoder(
            batch.evidence_type_ids,
            evidence_repr,
            batch.evidence_mask,
            preprojected=True,
        )

    def encode_observation(
        self,
        batch: WorkflowWorldModelBatch,
        sample_latent: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # 把当前观测编码为 posterior latent。
        # 输入包括 task、workflow state、历史 steps、证据集合、预算、协作图和上一步 hidden state。
        # 返回 latent、posterior 均值方差、下一 hidden state 和 graph 表示。
        sample_latent = self.training if sample_latent is None else sample_latent
        task_repr = self._encode_task_inputs(batch)
        task_type_repr = self.task_type_embedding(batch.task_type_ids)
        workflow_repr = self.workflow_state_embedding(batch.workflow_state_ids)
        step_repr = self.step_encoder(
            batch.step_role_ids,
            batch.step_action_ids,
            batch.step_features,
            batch.step_text_field_type_ids,
            batch.step_text_field_features,
            batch.step_text_field_mask,
            batch.step_mask,
        )
        evidence_repr = self._encode_evidence_inputs(batch)
        budget_repr = self.budget_projection(batch.budget_features)
        graph_repr = self.graph_encoder(
            batch.graph_node_ids,
            batch.graph_node_features,
            batch.graph_adj,
            batch.graph_mask,
        )
        hidden_state = batch.hidden_state
        if hidden_state is None:
            hidden_state = task_repr.new_zeros(batch.batch_size, self.config.hidden_dim)
        fused = self.observation_fusion(
            torch.cat(
                [
                    task_repr,
                    task_type_repr,
                    workflow_repr,
                    step_repr,
                    evidence_repr,
                    budget_repr,
                    graph_repr,
                    hidden_state,
                ],
                dim=-1,
            )
        )
        posterior_features, next_hidden = self.observation_state_cell(fused, hidden_state)
        post_mean, post_logvar = self.posterior_head(posterior_features).chunk(2, dim=-1)
        post_logvar = post_logvar.clamp(
            min=self.config.latent_logvar_min,
            max=self.config.latent_logvar_max,
        )
        latent = self._sample_latent(post_mean, post_logvar, sample=sample_latent)
        return latent, post_mean, post_logvar, next_hidden, graph_repr

    def checkpoint_state_dict(self) -> Dict[str, Tensor]:
        state = self.state_dict()
        if self.config.use_llm_text_encoder and self.config.text_encoder_freeze:
            return {name: value for name, value in state.items() if not name.startswith("text_encoder.")}
        return state

    def forward(self, batch: WorkflowWorldModelBatch, sample_latent: Optional[bool] = None) -> WorkflowWorldModelOutput:
        latent, post_mean, post_logvar, hidden_state, graph_repr = self.encode_observation(
            batch,
            sample_latent=sample_latent,
        )
        action_repr = self.encode_action(batch.action_kind_ids, batch.action_name_ids, batch.action_features)
        # 根据当前 latent、动作编码和图编码预测下一步 prior。
        transition_input = self.transition_input(torch.cat([latent, action_repr, graph_repr], dim=-1))
        prior_features, prior_hidden = self.transition_cell(transition_input, hidden_state)
        prior_mean, prior_logvar = self.prior_head(torch.cat([prior_features, graph_repr], dim=-1)).chunk(2, dim=-1)
        prior_logvar = prior_logvar.clamp(
            min=self.config.latent_logvar_min,
            max=self.config.latent_logvar_max,
        )
        head_input = torch.cat([prior_mean, graph_repr, prior_features], dim=-1)
        # 辅助头提供更密集的中间监督，例如 coverage / conflict / readiness。
        aux = {name: head(head_input).squeeze(-1) for name, head in self.aux_heads.items()}
        reward, reward_logits = self._decode_reward_output(self.reward_head(head_input))
        value, value_logits = self._decode_value_output(self.value_head(head_input))
        return WorkflowWorldModelOutput(
            latent=latent,
            latent_mean=post_mean,
            latent_logvar=post_logvar,
            hidden_state=hidden_state,
            graph_embedding=graph_repr,
            action_embedding=action_repr,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            prior_hidden_state=prior_hidden,
            reward=reward,
            reward_logits=reward_logits,
            cost=self._cost_output(self.cost_head(head_input).squeeze(-1)),
            done_logits=self.done_head(head_input).squeeze(-1),
            value=value,
            value_logits=value_logits,
            uncertainty=F.softplus(self.uncertainty_head(head_input).squeeze(-1)),
            valid_action_logits=self.valid_head(torch.cat([prior_mean, graph_repr], dim=-1)),
            aux=aux,
        )

    def imagine_rollout(
        self,
        latent: Tensor,
        hidden_state: Tensor,
        graph_embedding: Tensor,
        action_embeddings: Sequence[Tensor],
        gamma: float = 0.99,
    ) -> List[Dict[str, Tensor]]:
        # 在 latent 空间里做 imagined rollout，给 planner/reranker 提供多步估计。
        rollout: List[Dict[str, Tensor]] = []
        current_latent = latent
        current_hidden = hidden_state
        running_return = latent.new_zeros(latent.shape[0])
        for step_index, action_repr in enumerate(action_embeddings):
            transition_input = self.transition_input(torch.cat([current_latent, action_repr, graph_embedding], dim=-1))
            prior_features, current_hidden = self.transition_cell(transition_input, current_hidden)
            prior_mean, prior_logvar = self.prior_head(torch.cat([prior_features, graph_embedding], dim=-1)).chunk(2, dim=-1)
            head_input = torch.cat([prior_mean, graph_embedding, prior_features], dim=-1)
            reward, _ = self._decode_reward_output(self.reward_head(head_input))
            cost = self._cost_output(self.cost_head(head_input).squeeze(-1))
            uncertainty = F.softplus(self.uncertainty_head(head_input).squeeze(-1))
            value, _ = self._decode_value_output(self.value_head(head_input))
            running_return = running_return + (gamma**step_index) * (reward - cost)
            rollout.append(
                {
                    "prior_mean": prior_mean,
                    "prior_logvar": prior_logvar,
                    "hidden_state": current_hidden,
                    "reward": reward,
                    "cost": cost,
                    "uncertainty": uncertainty,
                    "value": value,
                    "return": running_return + (gamma ** (step_index + 1)) * value,
                }
            )
            current_latent = prior_mean
        return rollout

    def q_value(self, output: WorkflowWorldModelOutput, gamma: float = 0.99) -> Tensor:
        # 使用 reward - cost + gamma * value 近似一步 Q 值。
        return output.reward - output.cost + gamma * output.value

    def compute_losses(
        self,
        batch: WorkflowWorldModelBatch,
        next_batch: Optional[WorkflowWorldModelBatch] = None,
        output: Optional[WorkflowWorldModelOutput] = None,
        counterfactual_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # 汇总世界模型训练所需的所有损失项。
        output = self.forward(batch) if output is None else output
        total = output.reward.new_zeros(())
        losses: Dict[str, Tensor] = {"total": total}
        weights = self.config.loss_weights

        if next_batch is not None:
            next_mean, next_logvar = self._encode_transition_target(next_batch)
            losses["latent"] = self._latent_alignment_loss(output.prior_mean, next_mean)
            if float(self.config.latent_logvar_weight) > 0:
                losses["latent_logvar"] = F.smooth_l1_loss(output.prior_logvar, next_logvar)
            losses["kl"] = self._reduce_kl_loss(
                _gaussian_kl(
                    next_mean,
                    next_logvar,
                    output.prior_mean,
                    output.prior_logvar,
                )
            )
            losses["total"] = losses["total"] + weights["latent"] * losses["latent"] + weights["kl"] * losses["kl"]
            if "latent_logvar" in losses:
                losses["total"] = losses["total"] + float(self.config.latent_logvar_weight) * losses["latent_logvar"]

        targets = batch.targets
        if targets is None:
            return losses

        if targets.reward is not None:
            if self.config.use_reward_buckets and output.reward_logits is not None:
                reward_targets = self._reward_bucket_targets(targets.reward)
                smoothing = float(self.config.reward_bucket_label_smoothing)
                losses["reward"] = F.cross_entropy(output.reward_logits, reward_targets, label_smoothing=smoothing)
            else:
                losses["reward"] = F.smooth_l1_loss(output.reward, targets.reward)
            losses["total"] = losses["total"] + weights["reward"] * losses["reward"]
        if targets.cost is not None:
            losses["cost"] = F.smooth_l1_loss(output.cost, targets.cost)
            losses["total"] = losses["total"] + weights["cost"] * losses["cost"]
        if targets.done is not None:
            losses["done"] = F.binary_cross_entropy_with_logits(output.done_logits, targets.done.float())
            losses["total"] = losses["total"] + weights["done"] * losses["done"]
        if targets.value is not None:
            if self.config.use_value_buckets and output.value_logits is not None:
                value_targets = self._value_bucket_targets(targets.value)
                smoothing = float(self.config.value_bucket_label_smoothing)
                losses["value"] = F.cross_entropy(output.value_logits, value_targets, label_smoothing=smoothing)
            else:
                losses["value"] = F.smooth_l1_loss(output.value, targets.value)
            losses["total"] = losses["total"] + weights["value"] * losses["value"]
        if targets.uncertainty is not None:
            losses["uncertainty"] = F.smooth_l1_loss(output.uncertainty, targets.uncertainty)
            losses["total"] = losses["total"] + weights["uncertainty"] * losses["uncertainty"]
        if targets.next_valid_mask is not None:
            # valid_action_mask 作为多标签监督，约束模型不要给出明显无效的下一步动作。
            losses["valid"] = F.binary_cross_entropy_with_logits(
                output.valid_action_logits,
                targets.next_valid_mask.float(),
            )
            losses["total"] = losses["total"] + weights["valid"] * losses["valid"]
        if targets.aux:
            aux_terms = [F.smooth_l1_loss(output.aux[name], target) for name, target in targets.aux.items() if name in output.aux]
            if aux_terms:
                losses["aux"] = torch.stack(aux_terms).mean()
                losses["total"] = losses["total"] + weights["aux"] * losses["aux"]

        counterfactual_targets = targets.counterfactual_credit if counterfactual_targets is None else counterfactual_targets
        if counterfactual_targets is not None and float(weights.get("counterfactual", 0.0)) > 0:
            # 反事实监督同时做数值拟合和排序拟合，方便后续做 rerank/credit。
            q_values = self.q_value(output)
            cf_reg = F.smooth_l1_loss(q_values, counterfactual_targets)
            cf_rank = _pairwise_ranking_loss(q_values, counterfactual_targets)
            losses["counterfactual"] = cf_reg + cf_rank
            losses["total"] = losses["total"] + weights["counterfactual"] * losses["counterfactual"]
        return losses


class WorkflowStateAdapter:
    # 把 recorder JSONL 结构化记录转换成定长张量 batch。
    # 设计原则是只使用当前 step 可观测的信息，不直接读入 gold answer 等泄漏字段。
    def __init__(
        self,
        agent_roles: Sequence[str],
        action_names: Optional[Sequence[str]] = None,
        config: Optional[WorkflowWorldModelConfig] = None,
    ) -> None:
        self.config = config or WorkflowWorldModelConfig()
        self.role_to_id = {name: idx + 1 for idx, name in enumerate(agent_roles)}
        actions = list(action_names) if action_names is not None else list(agent_roles)
        self.action_to_id = {name: idx + 1 for idx, name in enumerate(actions)}
        self.task_type_to_id = {"unknown": 1}
        self.workflow_state_to_id = {"unknown": 1}
        self.action_kind_to_id = {"unknown": 0, "primitive": 1, "macro": 2, "mutation": 3}
        self.step_text_field_to_id = {
            "parameter": 1,
            "answer_summary": 2,
            "step_data_summary": 3,
            "raw_prompt": 4,
            "raw_response": 5,
            "system_prompt": 6,
        }
        self.vocab_frozen = False
        self.qwen_text_encoder: Optional[QwenTextEmbeddingEncoder] = None
        self.text_tokenizer = None
        self._text_token_cache: Dict[Tuple[str, int], Tuple[List[int], List[int]]] = {}
        self.config.step_dim = self.config.step_numeric_dim + self.config.step_text_field_dim
        if self.config.use_qwen_text_encoder:
            self.qwen_text_encoder = QwenTextEmbeddingEncoder(
                model_name=self.config.qwen_text_encoder_model_name,
                batch_size=self.config.qwen_text_encoder_batch_size,
                devices=self.config.qwen_text_encoder_devices,
                cache_path=self.config.qwen_text_cache_path,
            )
            embedding_dim = int(self.qwen_text_encoder.embedding_dim)
            self.config.task_dim = embedding_dim
            self.config.evidence_dim = embedding_dim
            self.config.step_text_field_dim = embedding_dim
            self.config.step_dim = self.config.step_numeric_dim + embedding_dim

    def _id(self, mapping: Dict[str, int], key: object) -> int:
        # 词表冻结后，新 token 统一回退到 unknown；未冻结时则动态扩展。
        key = str(key or "unknown").strip() or "unknown"
        if key not in mapping:
            if self.vocab_frozen:
                return mapping.get("unknown", 1)
            mapping[key] = len(mapping) + 1
        return mapping[key]

    def _text_features(self, text: str, dim: int) -> List[float]:
        # 这里先用轻量文本统计特征占位，后续可替换成更强的文本编码器。
        text = str(text or "")
        length = min(len(text), 4096)
        words = text.split()
        word_count = len(words)
        digit_count = sum(ch.isdigit() for ch in text)
        punct_count = sum(ch in ".,;:!?-" for ch in text)
        url_count = text.count("http")
        unique_ratio = len(set(words)) / max(word_count, 1)
        features = [
            length / 4096.0,
            min(word_count, 512) / 512.0,
            digit_count / max(length, 1),
            punct_count / max(length, 1),
            min(url_count, 4) / 4.0,
            unique_ratio,
            1.0 if "?" in text else 0.0,
            1.0 if "\n" in text else 0.0,
        ]
        return (features + [0.0] * dim)[:dim]

    def _encode_text_feature(self, text: str, dim: int, prompt_name: Optional[str] = None) -> List[float]:
        if self.qwen_text_encoder is not None:
            return self.qwen_text_encoder.encode([str(text or "")], prompt_name=prompt_name)[0][:dim]
        return self._text_features(text, dim)

    def _step_text_fields(self, step: Dict[str, Any]) -> List[Tuple[str, str]]:
        return [
            ("parameter", str(step.get("parameter", ""))),
            ("answer_summary", str(step.get("answer_summary", ""))),
            ("step_data_summary", str(step.get("step_data_summary", ""))),
            ("raw_prompt", str(step.get("raw_prompt", ""))),
            ("raw_response", str(step.get("raw_response", ""))),
            ("system_prompt", str(step.get("system_prompt", ""))),
        ]

    def _encode_step_text_bundle(self, step: Dict[str, Any]) -> List[Tuple[str, List[float], float]]:
        fields = self._step_text_fields(step)
        if self.qwen_text_encoder is not None:
            encoded = self.qwen_text_encoder.encode([text for _, text in fields])
        else:
            encoded = [self._text_features(text, self.config.step_text_field_dim) for _, text in fields]
        results: List[Tuple[str, List[float], float]] = []
        for (field_name, text), embedding in zip(fields, encoded):
            field_mask = 1.0 if str(text).strip() else 0.0
            values = list(embedding)[: self.config.step_text_field_dim]
            if len(values) < self.config.step_text_field_dim:
                values = values + [0.0] * (self.config.step_text_field_dim - len(values))
            results.append((field_name, values, field_mask))
        return results

    def _ensure_text_tokenizer(self) -> None:
        if not self.config.use_llm_text_encoder or self.text_tokenizer is not None:
            return
        if not self.config.text_encoder_model_path:
            raise ValueError("text_encoder_model_path must be set when LLM text encoding is enabled.")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required when --use-llm-text-encoder is enabled.") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_encoder_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            raise ValueError("The selected tokenizer does not define eos_token or unk_token for padding fallback.")
        self.text_tokenizer = tokenizer

    def _tokenize_text(self, text: str, max_length: int) -> Tuple[List[int], List[int]]:
        self._ensure_text_tokenizer()
        key = (str(text or ""), int(max_length))
        cached = self._text_token_cache.get(key)
        if cached is not None:
            return cached
        if self.text_tokenizer is None:
            raise RuntimeError("Text tokenizer is not initialized.")
        encoded = self.text_tokenizer(
            key[0],
            padding="max_length",
            truncation=True,
            max_length=key[1],
            return_attention_mask=True,
        )
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        attention_mask = [int(token_id) for token_id in encoded["attention_mask"]]
        cached = (input_ids, attention_mask)
        self._text_token_cache[key] = cached
        return cached

    def _task_question_text(self, task: Dict) -> str:
        # Recorder 里可能保留 Answer 等 gold 字段，但世界模型只读取可观测题面。
        return str(task.get("question", task.get("Question", "")))

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def _normalize_tokens(self, value: object, clip: float = 200000.0) -> float:
        raw = max(self._safe_float(value, 0.0), 0.0)
        return min(torch.log1p(torch.tensor(raw)).item() / torch.log1p(torch.tensor(clip)).item(), 1.0)

    def _normalize_cost(self, value: object, clip: float = 1.0e8) -> float:
        raw = max(self._safe_float(value, 0.0), 0.0)
        return min(torch.log1p(torch.tensor(raw)).item() / torch.log1p(torch.tensor(clip)).item(), 1.0)

    def _scan(self, records: Sequence[Dict]) -> None:
        # 先扫描一遍数据，补全 role/action/task_type/workflow_state 词表。
        for record in records:
            task = record.get("task", {})
            state = record.get("state", {})
            self._id(self.task_type_to_id, task.get("task_type", task.get("type", "unknown")))
            self._id(self.workflow_state_to_id, state.get("workflow_state", state.get("state", "unknown")))
            for step in state.get("executed_steps", []):
                self._id(self.role_to_id, step.get("agent", "unknown"))
                self._id(self.action_to_id, step.get("action", "unknown"))
            for node in record.get("graph", {}).get("nodes", []):
                self._id(self.role_to_id, node)
            action = record.get("action", {})
            self._id(self.action_to_id, action.get("name", "unknown"))
            for valid_action in record.get("next_state_targets", {}).get("valid_action_mask", state.get("valid_actions", [])):
                self._id(self.action_to_id, valid_action)

    def scan_records(self, records: Sequence[Dict]) -> None:
        self._scan(records)

    def freeze_vocab(self) -> None:
        self.vocab_frozen = True

    def build_batch(
        self,
        records: Sequence[Dict],
        hidden_state: Optional[Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> WorkflowWorldModelBatch:
        cfg = self.config
        batch_size = len(records)
        valid_dim = len(self.action_to_id)

        # 为固定形状 batch 预先分配所有输入张量。
        task_features = torch.zeros(batch_size, cfg.task_dim)
        task_type_ids = torch.zeros(batch_size, dtype=torch.long)
        workflow_state_ids = torch.zeros(batch_size, dtype=torch.long)
        step_role_ids = torch.zeros(batch_size, cfg.max_steps, dtype=torch.long)
        step_action_ids = torch.zeros(batch_size, cfg.max_steps, dtype=torch.long)
        step_features = torch.zeros(batch_size, cfg.max_steps, cfg.step_numeric_dim)
        step_text_field_type_ids = torch.zeros(batch_size, cfg.max_steps, cfg.num_step_text_fields, dtype=torch.long)
        step_text_field_features = torch.zeros(
            batch_size,
            cfg.max_steps,
            cfg.num_step_text_fields,
            cfg.step_text_field_dim,
        )
        step_text_field_mask = torch.zeros(batch_size, cfg.max_steps, cfg.num_step_text_fields)
        step_mask = torch.zeros(batch_size, cfg.max_steps)
        evidence_type_ids = torch.zeros(batch_size, cfg.max_evidence, dtype=torch.long)
        evidence_features = torch.zeros(batch_size, cfg.max_evidence, cfg.evidence_dim)
        evidence_mask = torch.zeros(batch_size, cfg.max_evidence)
        task_text_input_ids = None
        task_text_attention_mask = None
        evidence_text_input_ids = None
        evidence_text_attention_mask = None
        if cfg.use_llm_text_encoder:
            task_text_input_ids = torch.zeros(batch_size, cfg.task_text_max_length, dtype=torch.long)
            task_text_attention_mask = torch.zeros(batch_size, cfg.task_text_max_length, dtype=torch.long)
            evidence_text_input_ids = torch.zeros(
                batch_size,
                cfg.max_evidence,
                cfg.evidence_text_max_length,
                dtype=torch.long,
            )
            evidence_text_attention_mask = torch.zeros(
                batch_size,
                cfg.max_evidence,
                cfg.evidence_text_max_length,
                dtype=torch.long,
            )
        budget_features = torch.zeros(batch_size, cfg.budget_dim)
        graph_node_ids = torch.zeros(batch_size, cfg.max_nodes, dtype=torch.long)
        graph_node_features = torch.zeros(batch_size, cfg.max_nodes, cfg.node_dim)
        graph_adj = torch.zeros(batch_size, cfg.max_nodes, cfg.max_nodes)
        graph_mask = torch.zeros(batch_size, cfg.max_nodes)
        action_kind_ids = torch.zeros(batch_size, dtype=torch.long)
        action_name_ids = torch.zeros(batch_size, dtype=torch.long)
        action_features = torch.zeros(batch_size, cfg.action_dim)
        reward = torch.zeros(batch_size)
        cost = torch.zeros(batch_size)
        done = torch.zeros(batch_size)
        value = torch.zeros(batch_size)
        uncertainty = torch.zeros(batch_size)
        next_valid_mask = torch.zeros(batch_size, valid_dim)
        aux = {name: torch.zeros(batch_size) for name in cfg.aux_names}
        counterfactual_credit = torch.zeros(batch_size)

        evidence_kind = {"reasoning": 1, "tool": 2, "answer": 3}
        for row, record in enumerate(records):
            # task/state/action/outcome/returns/credit_targets 都在这里被转成张量。
            task = record.get("task", {})
            state = record.get("state", {})
            steps = list(state.get("executed_steps", []))[-cfg.max_steps:]
            question = self._task_question_text(task)
            if cfg.use_llm_text_encoder:
                input_ids, attention_mask = self._tokenize_text(question, cfg.task_text_max_length)
                if task_text_input_ids is None or task_text_attention_mask is None:
                    raise RuntimeError("Task text tensors were not initialized.")
                task_text_input_ids[row] = torch.tensor(input_ids, dtype=torch.long)
                task_text_attention_mask[row] = torch.tensor(attention_mask, dtype=torch.long)
            else:
                task_features[row] = torch.tensor(
                    self._encode_text_feature(
                        question,
                        cfg.task_dim,
                    )
                )
            task_type_ids[row] = self._id(self.task_type_to_id, task.get("task_type", task.get("type", "unknown")))
            workflow_state_ids[row] = self._id(
                self.workflow_state_to_id,
                state.get("workflow_state", state.get("state", "unknown")),
            )

            # 历史 steps 使用右对齐填充，保留最近的执行轨迹。
            for offset, step in enumerate(steps, start=cfg.max_steps - len(steps)):
                step_role_ids[row, offset] = self._id(self.role_to_id, step.get("agent", "unknown"))
                step_action_ids[row, offset] = self._id(self.action_to_id, step.get("action", "unknown"))
                numeric_features = [
                    1.0 if bool(step.get("success", False)) else 0.0,
                    self._normalize_tokens(step.get("tokens", 0.0), clip=200000.0),
                    self._normalize_cost(step.get("cost", 0.0), clip=1.0e8),
                    min(len(str(step.get("parameter", ""))), 512) / 512.0,
                    min(len(str(step.get("answer_summary", ""))), 512) / 512.0,
                    min(len(str(step.get("step_data_summary", ""))), 1024) / 1024.0,
                    1.0 if step.get("answer_summary") else 0.0,
                    1.0 if step.get("step_data_summary") else 0.0,
                ]
                step_features[row, offset] = torch.tensor(numeric_features[: cfg.step_numeric_dim])
                for field_index, (field_name, embedding, field_mask) in enumerate(self._encode_step_text_bundle(step)):
                    if field_index >= cfg.num_step_text_fields:
                        break
                    step_text_field_type_ids[row, offset, field_index] = self.step_text_field_to_id.get(field_name, 0)
                    step_text_field_features[row, offset, field_index] = torch.tensor(embedding[: cfg.step_text_field_dim])
                    step_text_field_mask[row, offset, field_index] = field_mask
                step_mask[row, offset] = 1.0

            evidence_items: List[Tuple[str, str]] = []
            # reasoning/tool/answer 统一视作证据集合。
            evidence_items += [("reasoning", str(item)) for item in state.get("reasoning_results", [])]
            evidence_items += [("tool", str(item)) for item in state.get("tool_results", [])]
            evidence_items += [("answer", str(item)) for item in state.get("recent_answers", [])]
            evidence_items = evidence_items[-cfg.max_evidence :]
            for offset, (kind, text) in enumerate(evidence_items, start=cfg.max_evidence - len(evidence_items)):
                evidence_type_ids[row, offset] = evidence_kind[kind]
                if cfg.use_llm_text_encoder:
                    input_ids, attention_mask = self._tokenize_text(text, cfg.evidence_text_max_length)
                    if evidence_text_input_ids is None or evidence_text_attention_mask is None:
                        raise RuntimeError("Evidence text tensors were not initialized.")
                    evidence_text_input_ids[row, offset] = torch.tensor(input_ids, dtype=torch.long)
                    evidence_text_attention_mask[row, offset] = torch.tensor(attention_mask, dtype=torch.long)
                else:
                    evidence_features[row, offset] = torch.tensor(
                        self._encode_text_feature(text, cfg.evidence_dim)
                    )
                evidence_mask[row, offset] = 1.0

            budget = state.get("budget", {})
            constraints = task.get("constraints", {})
            if not isinstance(constraints, dict):
                constraints = {}
            budget_features[row] = torch.tensor(
                [
                    min(self._safe_float(budget.get("step_index", len(steps))), 32.0) / 32.0,
                    self._normalize_tokens(budget.get("used_tokens", record.get("total_tokens", 0.0)), clip=500000.0),
                    self._normalize_cost(budget.get("used_cost", record.get("total_cost", 0.0)), clip=1.0e8),
                    min(self._safe_float(constraints.get("budget", 1.0), 1.0), 1.0),
                ][: cfg.budget_dim]
            )

            graph = record.get("graph", {})
            # graph 提供当前 agent 生态信息，包括节点统计和边结构。
            nodes = list(graph.get("nodes", self.role_to_id.keys()))[: cfg.max_nodes]
            node_stats = graph.get("node_stats", {})
            node_slot = {name: idx for idx, name in enumerate(nodes)}
            for idx, name in enumerate(nodes):
                stats = node_stats.get(name, {})
                graph_node_ids[row, idx] = self._id(self.role_to_id, name)
                graph_node_features[row, idx] = torch.tensor(
                    [
                        self._safe_float(stats.get("success_rate", 0.0)),
                        self._normalize_cost(stats.get("avg_cost", 0.0), clip=1.0e8),
                        self._safe_float(stats.get("avg_credit", 0.0)),
                        min(self._safe_float(stats.get("usage_count", 0.0)), 100.0) / 100.0,
                        1.0 if name in {"TavilyAgent", "WebsiteAgent", "ArxivAgent"} else 0.0,
                        1.0 if name == "TerminatorAgent" else 0.0,
                    ][: cfg.node_dim]
                )
                graph_mask[row, idx] = 1.0
                graph_adj[row, idx, idx] = 1.0
            for src, dst in graph.get("edges", []):
                if src in node_slot and dst in node_slot:
                    graph_adj[row, node_slot[src], node_slot[dst]] = 1.0

            action = record.get("action", {})
            # 当前 action 是 transition 的条件输入。
            action_kind_ids[row] = self.action_kind_to_id.get(str(action.get("kind", "primitive")), 0)
            action_name_ids[row] = self._id(self.action_to_id, action.get("name", "unknown"))
            action_name = str(action.get("name", ""))
            action_features[row] = torch.tensor(
                [
                    1.0 if action.get("kind", "primitive") == "macro" else 0.0,
                    1.0 if action.get("kind", "primitive") == "mutation" else 0.0,
                    1.0 if "Terminator" in action_name else 0.0,
                    1.0 if action_name in {"TavilyAgent", "WebsiteAgent", "ArxivAgent"} else 0.0,
                    min(len(action_name), 64) / 64.0,
                    self._normalize_cost(action.get("estimated_cost", 0.0), clip=1.0e8),
                ][: cfg.action_dim]
            )

            outcome = record.get("outcome", {})
            next_state_targets = record.get("next_state_targets", {})
            # 监督目标主要来自 recorder 记录的真实执行结果。
            reward[row] = self._safe_float(outcome.get("reward", 0.0))
            cost[row] = self._normalize_cost(outcome.get("cost_delta", 0.0), clip=1.0e8)
            done[row] = 1.0 if bool(outcome.get("done", False)) else 0.0
            value[row] = self._safe_float(record.get("returns", {}).get("mc_return", outcome.get("reward", 0.0)))
            uncertainty[row] = self._safe_float(next_state_targets.get("conflict_score", 0.0))
            for name in cfg.aux_names:
                aux[name][row] = self._safe_float(next_state_targets.get(name, 0.0))
            for valid_action in next_state_targets.get("valid_action_mask", state.get("valid_actions", [])):
                next_valid_mask[row, self._id(self.action_to_id, valid_action) - 1] = 1.0
            counterfactual_credit[row] = self._safe_float(
                record.get("credit_targets", {}).get("leave_one_out_gap", outcome.get("reward", 0.0))
            )

        batch = WorkflowWorldModelBatch(
            task_features=task_features,
            task_type_ids=task_type_ids,
            workflow_state_ids=workflow_state_ids,
            step_role_ids=step_role_ids,
            step_action_ids=step_action_ids,
            step_features=step_features,
            step_text_field_type_ids=step_text_field_type_ids,
            step_text_field_features=step_text_field_features,
            step_text_field_mask=step_text_field_mask,
            step_mask=step_mask,
            evidence_type_ids=evidence_type_ids,
            evidence_features=evidence_features,
            evidence_mask=evidence_mask,
            task_text_input_ids=task_text_input_ids,
            task_text_attention_mask=task_text_attention_mask,
            evidence_text_input_ids=evidence_text_input_ids,
            evidence_text_attention_mask=evidence_text_attention_mask,
            budget_features=budget_features,
            graph_node_ids=graph_node_ids,
            graph_node_features=graph_node_features,
            graph_adj=graph_adj,
            graph_mask=graph_mask,
            action_kind_ids=action_kind_ids,
            action_name_ids=action_name_ids,
            action_features=action_features,
            hidden_state=hidden_state,
            targets=WorkflowWorldModelTargets(
                reward=reward,
                cost=cost,
                done=done,
                value=value,
                uncertainty=uncertainty,
                next_valid_mask=next_valid_mask,
                aux=aux,
                counterfactual_credit=counterfactual_credit,
            ),
        )
        return batch if device is None else batch.to(device)
