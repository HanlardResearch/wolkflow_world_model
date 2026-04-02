import torch
import atexit
import os
import datetime
import json
import numpy as np
import torch.nn as nn
import yaml
import logging
from utils.other_utils import Singleton
from inference.policy.base_policy import LLMPolicy, LearningPolicy
from model.embedding import RewardModelTokenRepresentation
MODEL_WEIGHT_PATH="/extrahome0/HF_models/Qwen/Qwen3.5-4B"
global_config = yaml.safe_load(open("./config/global.yaml", "r"))
logger = logging.getLogger("train")

import os
import json
import torch
import torch.nn as nn
import logging
import re
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference.policy.base_policy import LLMPolicy, LearningPolicy
from model.model_utils import model_log_and_print
import yaml

global_config = yaml.safe_load(open("./config/global.yaml", "r"))
logger = logging.getLogger("train")


class LLMJSONPolicyParser:
    """解析LLM输出的JSON格式策略"""

    ACTION_PROB_KEY = "action_probabilities"
    FALLBACK_UNIFORM = True  # 解析失败时是否使用均匀分布

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict]:
        """从LLM输出中提取JSON对象"""
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取```json```块
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取最外层{}
        brace_pattern = r'(\{[\s\S]*\})'
        match = re.search(brace_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def parse_action_probs(
            json_data: Dict,
            action_names: List[str],
            action_dim: int
    ) -> torch.Tensor:
        """
        解析JSON中的动作概率，返回归一化的概率张量
        支持多种JSON格式:
        - {"action_probabilities": {"search": 0.3, "end": 0.7}}
        - {"search": 0.3, "end": 0.7}
        - [0.1, 0.2, ..., 0.7] (按action_names顺序)
        """
        probs = torch.zeros(action_dim, dtype=torch.float32)

        # 格式1: 嵌套在action_probabilities键中
        if LLMJSONPolicyParser.ACTION_PROB_KEY in json_data:
            prob_dict = json_data[LLMJSONPolicyParser.ACTION_PROB_KEY]
        # 格式2: 直接是动作名->概率的映射
        elif isinstance(json_data, dict) and any(k in action_names for k in json_data.keys()):
            prob_dict = json_data
        # 格式3: 概率列表（按action_names顺序）
        elif isinstance(json_data, list) and len(json_data) == action_dim:
            probs = torch.tensor(json_data, dtype=torch.float32)
            return torch.softmax(probs, dim=0)  # 确保归一化
        else:
            raise ValueError(f"Unsupported JSON format: {json_data}")

        # 解析动作名->概率映射
        valid_count = 0
        for idx, action_name in enumerate(action_names):
            if action_name in prob_dict:
                try:
                    probs[idx] = float(prob_dict[action_name])
                    valid_count += 1
                except (ValueError, TypeError):
                    continue

        # 如果没有有效概率，抛出异常触发fallback
        if valid_count == 0:
            raise ValueError("No valid action probabilities found in JSON")

        # 归一化（softmax确保数值稳定性）
        return torch.softmax(probs, dim=0)

    @classmethod
    def parse(cls, text: str, action_names: List[str], action_dim: int) -> torch.Tensor:
        """主解析入口，带fallback机制"""
        try:
            json_data = cls.extract_json_from_text(text)
            if json_data is None:
                raise ValueError("Failed to extract JSON from LLM output")
            return cls.parse_action_probs(json_data, action_names, action_dim)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}, using fallback distribution")
            if cls.FALLBACK_UNIFORM:
                # 均匀分布或基于prior的分布
                return torch.ones(action_dim, dtype=torch.float32) / action_dim
            raise


@Singleton
class ContinuousREINFORCE(LearningPolicy):
    def __init__(self, agent_graph, action_graph, config_path="config/policy.json"):
        super().__init__(agent_graph, action_graph)

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # ===== 设备与路径配置 =====
        self.device = self.config["device"]["type"]
        self.model_path = self.config["paths"]["model_path"]

        # ===== 训练参数 =====
        self.training = self.config["training"]["training"]
        self.loading = self.config["training"]["loading"]
        self.learning_rate = self.config["training"]["learning_rate"]
        self.gamma = self.config["training"]["gamma"]
        self.sample_size = self.config["training"]["sample_size"]
        self.lambda_kl_loss = self.config["training"]["lambda_kl_loss"]
        self.entropy_coef = self.config["training"].get("entropy_coef", 0.01)  # 熵正则系数

        # ===== Agent参数 =====
        self.max_num_agents = self.config["agent"]["max_num_agents"]
        self.next_num_agents = self.config["agent"]["next_num_agents"]
        self.max_path = self.config["agent"]["max_path"]
        self.threshold = self.config["agent"]["threshold"]

        # ===== LLM参数 =====
        self.llm_prior = self.config["llm"]["prior"]
        self.llm_prior_redistribution = self.config["llm"]["prior_redistribution"]
        self.redistribution_weight = self.config["llm"]["redistribution_weight"]

        # ===== 移除MLP，改用LLM直接生成策略 =====
        self.json_parser = LLMJSONPolicyParser()
        # 初始化LLM用于策略生成（可复用state_representation的模型，或独立加载）
        self.policy_llm = self._init_policy_llm()

        # 加载预训练模型（如需）
        if not self.training:
            self.load_model(self.get_latest_model_path())
        if self.loading:
            self.load_model(self.model_path)

        # ===== Agent映射 =====
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes
        self.action_names = [agent_graph.get_role_by_index(i) for i in range(self.actions_dim)]

        # ===== 轨迹追踪 =====
        self.executed_trajectories = []
        self.execution_count = 0
        self.current_trajectories = []
        self.current_trajectory_idx = 0

        # ===== 训练记录 =====
        self.policy_losses = []
        self.rewards_history = []
        self.action_probs_history = []
        self.llm_action_probs_history = []
        self.reward_from_rm = []
        self.accumulated_acc = []
        self.entropy_history = []

        # ===== 动作与奖励配置 =====
        self.end_action = torch.tensor(self.agent_graph.terminator_agent_index, device=self.device)
        self.web_actions = torch.tensor(self.agent_graph.search_agent_indices, device=self.device)

        reward_factors = self.config["agent"]["reward_factors"]
        self.agent_reward_factor = [reward_factors["default"]] * self.actions_dim
        self.agent_reward_factor[self.end_action.item()] = reward_factors["terminator"]
        for web_idx in self.web_actions:
            self.agent_reward_factor[web_idx.item()] = reward_factors["web_search"]

        # ===== 任务状态 =====
        self.current_task = None
        self.previous_task = None
        self.global_step = 0
        self.prob_step = 0

        # ===== 优化器（如需训练LLM参数，使用LoRA等参数高效方法） =====
        self.optimizer = self._init_optimizer()
        self.max_step_num = global_config.get("graph", {}).get("max_step_num", 20)
        self.llm_policy = LLMPolicy(self.agent_graph, self.action_graph)

        atexit.register(self.save_model)

    def _init_policy_llm(self):
        """初始化用于生成策略的LLM"""
        # 方案1: 复用state_representation的模型（如果架构兼容）
        # 方案2: 独立加载专门用于策略生成的模型
        model_name = self.config.get("policy_llm", {}).get("model_name", "Qwen3.5-4B")
        model_path = self.config.get("policy_llm", {}).get("model_path", MODEL_WEIGHT_PATH)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.truncation_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        model.eval()  # 推理模式，训练时再switch

        return {"model": model, "tokenizer": tokenizer}

    def _init_optimizer(self):
        """初始化优化器（仅在需要训练LLM参数时）"""
        if self.training and self.config.get("policy_llm", {}).get("trainable", False):
            # 建议使用参数高效微调（如LoRA）
            from peft import get_peft_model, LoraConfig
            peft_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1, task_type="CAUSAL_LM"
            )
            self.policy_llm["model"] = get_peft_model(self.policy_llm["model"], peft_config)
            return torch.optim.AdamW(
                self.policy_llm["model"].parameters(),
                lr=self.config.get("policy_llm", {}).get("lr", 1e-4)
            )
        return None  # 不训练时返回None

    def _build_policy_prompt(self, state_context: List[Dict], global_info) -> str:
        """构建LLM策略生成的prompt"""
        # 提取关键信息
        task_question = global_info.task.get("Question", "")
        current_state = global_info.workflow.state if hasattr(global_info, 'workflow') else "unknown"
        available_actions = ", ".join(self.action_names)

        # 构建few-shot示例（提升JSON输出稳定性）
        examples = [
            {
                "context": "用户问: 北京天气如何？",
                "available_actions": ["search", "end"],
                "output": {"action_probabilities": {"search": 0.8, "end": 0.2}}
            },
            {
                "context": "已获取搜索结果，需要总结",
                "available_actions": ["summarize", "search", "end"],
                "output": {"action_probabilities": {"summarize": 0.7, "search": 0.1, "end": 0.2}}
            }
        ]

        # 构建完整prompt
        prompt_parts = [
            "You are a multi-agent scheduler. Output action probabilities as JSON.\n",
            "Available actions: [" + available_actions + "]\n",
            "Output format: {\"action_probabilities\": {\"action_name\": probability, ...}}\n",
            "Probabilities must sum to 1.0\n\n"
        ]

        # 添加few-shot示例
        for ex in examples:
            prompt_parts.append(f"Example:\nContext: {ex['context']}\n")
            prompt_parts.append(f"Actions: {ex['available_actions']}\n")
            prompt_parts.append(f"Output: ```json\n{json.dumps(ex['output'])}\n```\n\n")

        # 添加当前任务
        prompt_parts.append(f"Current Task:\nContext: {task_question}\n")
        prompt_parts.append(f"Current State: {current_state}\n")
        prompt_parts.append("Output your action probabilities:\n")

        return "".join(prompt_parts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_policy_probs(self, state_context: List[Dict], global_info) -> torch.Tensor:
        """调用LLM生成动作概率分布"""
        prompt = self._build_policy_prompt(state_context, global_info)
        tokenizer = self.policy_llm["tokenizer"]
        model = self.policy_llm["model"]

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # 根据模型调整
        ).to(self.device)

        # Generate - 关键：限制生成长度，强制输出JSON
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # JSON通常不需要太长
                min_new_tokens=20,
                do_sample=True,
                temperature=0.7,  # 适度随机性
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                # 可选：使用logits_processor约束输出格式
            )

        # Decode & Parse
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        model_log_and_print(f"LLM Policy Output: {generated_text}")

        # 解析为概率分布
        action_probs = self.json_parser.parse(
            generated_text,
            action_names=self.action_names,
            action_dim=self.actions_dim
        )

        return action_probs.to(self.device)

    def _get_action_distribution(self, global_info) -> torch.Tensor:
        """
        核心改动：LLM直接处理原始对话历史，输出动作概率
        不再经过state embedding中间层
        """
        # 1. 获取原始对话历史（保持原有接口）
        role_list = global_info.agent_role_list()
        state_context = self.agent_graph.get_agent_dialog_history(
            role_list,
            question=global_info.task.get("Question")
        )

        # Qwen兼容处理（保持原有逻辑）
        if "qwen" in self.policy_llm["tokenizer"].__class__.__name__.lower():
            state_context = self.sanitize_messages_for_tokenizer(state_context)

        # 2. 【核心】LLM直接生成概率分布（替代: embedding → MLP → probs）
        with torch.set_grad_enabled(self.training and self.policy_llm.get("trainable", False)):
            action_probs = self._generate_policy_probs(state_context, global_info)

        # 3. LLM prior融合（保持原有逻辑）
        if self.llm_prior_redistribution and self.llm_prior:
            prior_action_probs = self.llm_policy.forward_prior(global_info)
            action_probs = (1 - self.redistribution_weight) * action_probs + \
                           self.redistribution_weight * prior_action_probs.to(self.device)
            action_probs = action_probs / (action_probs.sum() + 1e-10)

        return action_probs.to(self.device)  # shape: (action_dim,)

    def append_to_trajectory(self, trajectory_idx, agent_idx, prob_value,
                             global_info, prior_action_probs, m, rew=0):
        """
        改动: rew参数现在来自环境/执行结果，而非model预测
        如果当前步骤无法获取即时reward，可设为0，最终由终端奖励反向传播
        """
        # 计算步骤成本（保持原有逻辑）
        cost = self.logarithmic_cost(len(self.current_trajectories[trajectory_idx])) * \
               self.agent_reward_factor[agent_idx.item()]

        # ✅ rew现在来自环境反馈（如任务完成信号、外部评估等）
        # 如果当前步骤无即时奖励，rew=0，靠终端奖励+gamma传播
        step_reward = cost + rew  # 或根据需求调整组合方式

        self.current_trajectories[trajectory_idx].append({
            'prob': prob_value,
            'log_prob': m.log_prob(agent_idx),
            'state_identifier': global_info.workflow.state,
            'action': self.agent_role_list[agent_idx.item()],
            'reward': step_reward,  # ✅ 环境reward + 成本
            'reward_model': None,  # ❌ 不再使用model预测的reward
            'prior_prob': prior_action_probs[agent_idx.item()] if prior_action_probs is not None else None
        })

    def init_forward(self, global_info):
        print("\033[1;33mInit Policy Forward\033[0m")
        logger.info("[Init Policy Forward]")

        self.current_task = global_info.task
        self.update_executed_trajectories()

        # ✅ 直接获取动作分布（无state embedding中间层）
        action_probs = self._get_action_distribution(global_info)
        action_probs = action_probs.unsqueeze(0)  # (1, action_dim)

        # 记录历史
        self.action_probs_history.append(action_probs.T.squeeze(1))
        # ❌ 不再记录 reward_from_rm（因为移除了reward model）

        # 熵计算 & 采样（保持原有逻辑）
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
        self.entropy_history.append(entropy)
        m = torch.distributions.Categorical(action_probs)
        agent_indices = self.select_agents_by_threshold(action_probs).T.squeeze(1)

        # 轨迹初始化...
        self.current_trajectory_idx = 0
        length = len(self.current_trajectories) + agent_indices.shape[0]
        while len(self.current_trajectories) < length:
            self.current_trajectories.append([])

        for i, agent_idx in enumerate(agent_indices):
            prob_value = action_probs[0, agent_idx.item()]
            trajectory_idx = self.current_trajectory_idx if i == 0 else len(self.current_trajectories) - len(
                agent_indices) + i
            self.append_to_trajectory(trajectory_idx, agent_idx, prob_value, global_info, None, m, rew)

        return agent_indices

    def iter_forward(self, global_info):
        """
        迭代步骤的策略前向传播
        LLM直接根据对话历史生成动作概率分布，按概率采样获取下一步行动
        """
        print("\033[1;33mFollowing Policy Forward\033[0m")
        logger.info("Following Policy Forward")

        # ===== 1. 任务状态更新 =====
        self.current_task = global_info.task

        # ===== 2. 获取动作概率分布（核心改动：LLM直接生成，无中间编码层）=====
        action_probs = self._get_action_distribution(global_info)  # shape: (action_dim,)
        action_probs = action_probs.unsqueeze(0)  # -> (1, action_dim) 保持batch维度

        # ===== 3. LLM Prior融合（可选，保持原有逻辑）=====
        prior_action_probs = None
        if self.llm_prior:
            prior_action_probs = self.llm_policy.forward_prior(global_info)  # (action_dim,)
            print("Prior action probs: {}".format(prior_action_probs))
            logger.info("Prior action probs: {}".format(prior_action_probs))

            if self.llm_prior_redistribution and prior_action_probs is not None:
                # 线性插值融合 + 重归一化
                action_probs = (1 - self.redistribution_weight) * action_probs + \
                               self.redistribution_weight * prior_action_probs.to(self.device).unsqueeze(0)
                action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # ===== 4. 记录训练指标（移除reward_from_rm）=====
        self.action_probs_history.append(action_probs.T.squeeze(1))  # (action_dim,)
        # ❌ self.reward_from_rm.append(rew)  # 已移除state_representation，无需记录

        # 熵计算（用于熵正则化，鼓励探索）
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
        self.entropy_history.append(entropy)
        logger.info("Action probs: {}, Entropy: {:.4f}".format(action_probs, entropy.item()))

        # ===== 5. 按概率采样选择智能体（保持原有逻辑）=====
        m = torch.distributions.Categorical(action_probs)
        agent_indices = self.select_agents_by_threshold(action_probs).T.squeeze(1)  # (num_selected,)

        # ===== 6. 轨迹管理：初始化/扩展轨迹列表 =====
        self.current_trajectory_idx = global_info.path_id
        length = len(self.current_trajectories) + len(agent_indices) - 1
        original_length = len(self.current_trajectories)

        while len(self.current_trajectories) < length:
            self.current_trajectories.append([])

        # ===== 7. 为每个选中智能体记录轨迹（核心：reward来自环境/成本，非model预测）=====
        for i, agent_idx in enumerate(agent_indices):
            prob_value = action_probs[0, agent_idx.item()]  # 标量概率

            # 确定轨迹索引：第一个用current_trajectory_idx，后续新建
            if i == 0:
                trajectory_idx = self.current_trajectory_idx
            else:
                trajectory_idx = original_length + i - 1
                # 克隆已有轨迹作为新分支起点（保持多路径探索）
                self.current_trajectories[trajectory_idx] = self.clone_trajectory(self.current_trajectory_idx)

            # 记录单步决策到轨迹
            self.append_to_trajectory(
                trajectory_idx=trajectory_idx,
                agent_idx=agent_idx,
                prob_value=prob_value,
                global_info=global_info,
                prior_action_probs=prior_action_probs,
                distribution=m,
                env_reward=0.0  # ✅ 即时reward来自环境，当前步骤若无则设为0，靠终端奖励+gamma传播
            )

        print("Agent Indices: {}".format(agent_indices))
        return agent_indices


    def update(self):
        """
        REINFORCE策略更新 - 适配LLM直接生成策略的新架构
        核心逻辑保持不变: -logπ(a|s)·R + λ·KL + η·H
        """
        logger.info("Update")

        # ===== 1. 推理模式：仅记录指标，不更新 =====
        if not self.training:
            metrics = {
                'reasoning/action_probs': torch.sum(torch.stack(self.action_probs_history), dim=0).cpu().numpy(),
                'reasoning/entropy': np.mean([e.detach().cpu().item() for e in self.entropy_history]),
                'reasoning/num_trajectories': len(self.executed_trajectories),
            }
            logger.info("Inference metrics: {}".format(metrics))
            self._reset_buffers()
            return metrics

        # ===== 2. 等待足够样本 =====
        if len(self.executed_trajectories) < self.sample_size:
            logger.info(f"Waiting for samples: {len(self.executed_trajectories)}/{self.sample_size}")
            return {}

        logger.info(f"Update with sample_size={self.sample_size}, global_step={self.global_step}")

        # ===== 3. 批量计算: returns, loss terms, metrics =====
        episode_returns = []
        episode_lengths = []
        episode_acc = []
        episode_costs = []
        kl_losses = []
        policy_loss_terms = []

        # 遍历采样批次
        for traj_batch in self.executed_trajectories[:self.sample_size]:
            # 仅处理已完成的轨迹
            completed_trajs = [t for t in traj_batch if t and t[-1].get('finalized', False)]
            if not completed_trajs:
                continue

            for trajectory in completed_trajs:
                # (a) 计算折扣回报: R_t = Σ γ^(k-t) · r_k
                returns = self.calculate_returns(trajectory)  # List[Tensor], len=trajectory_length

                # 记录轨迹级指标
                episode_returns.append(returns[0].item())  # 首步回报代表整条轨迹
                episode_lengths.append(len(trajectory))
                episode_acc.append(1.0 if returns[0].item() > 0 else 0.0)  # 简单成功率
                episode_costs.append(trajectory[-1].get('total_cost', 0.0))

                # (b) 计算每步loss项
                for step_data, R in zip(trajectory, returns):
                    if step_data.get('prob') is None or step_data.get('log_prob') is None:
                        continue

                    prob = step_data['prob']
                    log_prob = step_data['log_prob']

                    # KL散度正则: D_KL(π_prior || π_current)
                    kl_loss = torch.tensor(0.0, device=self.device)
                    prior_prob = step_data.get('prior_prob')
                    if prior_prob is not None and prob > 1e-10:
                        # 稳定计算: prior·log(prior/prob)
                        kl_loss = prior_prob * torch.log((prior_prob + 1e-10) / (prob + 1e-10))
                        kl_loss = torch.clamp(kl_loss, min=0.0)  # KL非负
                    kl_losses.append(kl_loss)

                    # REINFORCE loss: -∇logπ·R + λ·KL
                    loss_term = -log_prob * R + self.lambda_kl_loss * kl_loss
                    policy_loss_terms.append(loss_term)

        # ===== 4. 空样本保护 =====
        if not policy_loss_terms:
            logger.warning("No valid loss terms, skipping update")
            self._reset_buffers()
            return {}

        # ===== 5. 聚合Loss + 熵正则 =====
        policy_loss = torch.stack(policy_loss_terms).mean()

        # 熵正则化: +η·H(π) 鼓励探索（注意: REINFORCE中是 -logπ·R，所以熵项为 -η·(-H) = +η·H）
        if self.entropy_coef > 0 and self.entropy_history:
            mean_entropy = torch.stack(self.entropy_history).mean()
            policy_loss = policy_loss - self.entropy_coef * mean_entropy  # 最大化熵 = 最小化 -熵
            logger.debug(f"Entropy bonus: {self.entropy_coef * mean_entropy.item():.4f}")

        # ===== 6. 反向传播（仅当LLM可训练时）=====
        if self.optimizer is not None and self.policy_llm.get("trainable", False):
            self.optimizer.zero_grad()
            policy_loss.backward()

            # 梯度裁剪：防止LLM训练不稳定
            max_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(
                self.policy_llm["model"].parameters(),
                max_norm=max_norm
            )

            self.optimizer.step()
            logger.info(f"✓ Policy updated: loss={policy_loss.item():.4f}")
        else:
            logger.info(f"✓ Policy evaluation only (trainable=False): loss={policy_loss.item():.4f}")

        # ===== 7. 记录完整指标 =====
        metrics = self._compute_metrics(
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            episode_acc=episode_acc,
            episode_costs=episode_costs,
            kl_losses=kl_losses,
            policy_loss=policy_loss.item()
        )
        logger.info("Training metrics: {}".format(metrics))

        # ===== 8. 清理缓冲 + 步数更新 =====
        self._reset_buffers()
        self.global_step += 1

        # 定期保存（可选）
        if self.global_step % self.config.get("training", {}).get("save_every", 100) == 0:
            self.save_model(tag=f"step_{self.global_step}")

        return metrics


    def _compute_metrics(self, episode_returns, episode_lengths, episode_acc,
                         episode_costs, kl_losses, policy_loss):
        """计算并格式化训练指标"""
        metrics = {
            # 策略性能
            'reasoning/mean_return': np.mean(episode_returns) if episode_returns else 0.0,
            'reasoning/std_return': np.std(episode_returns) if len(episode_returns) > 1 else 0.0,
            'reasoning/mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'reasoning/success_rate': np.mean(episode_acc) if episode_acc else 0.0,
            'reasoning/mean_cost': np.mean(episode_costs) if episode_costs else 0.0,

            # 策略分布
            'reasoning/action_probs': torch.sum(torch.stack(self.action_probs_history), dim=0).cpu().numpy(),
            'reasoning/mean_entropy': np.mean([e.detach().cpu().item() for e in self.entropy_history]),

            # 训练动态
            'training/policy_loss': policy_loss,
            'training/mean_kl_loss': np.mean([kl.item() for kl in kl_losses]) if kl_losses else 0.0,
            'training/global_step': self.global_step,
            'training/sample_size': len(episode_returns),
        }

        # 添加action维度的详细概率（可选，debug用）
        if self.config.get("debug", False):
            avg_probs = torch.stack(self.action_probs_history).mean(dim=0).cpu().numpy()
            for idx, (name, prob) in enumerate(zip(self.action_names, avg_probs)):
                metrics[f'debug/prob_{name}'] = prob

        return metrics


    def _reset_buffers(self):
        """清空训练缓冲，准备下一轮收集"""
        self.current_trajectories = []
        self.executed_trajectories = []
        self.action_probs_history = []
        self.entropy_history = []
        self.execution_count = 0
        # ❌ 已移除: self.reward_from_rm = []
        logger.debug("Buffers reset")