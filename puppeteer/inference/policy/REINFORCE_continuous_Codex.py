import datetime
import json
import logging
import os
import re

import torch
import yaml

from inference.policy.base_policy import LearningPolicy
from model import query_gpt
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.other_utils import Singleton

global_config = yaml.safe_load(open("./config/global.yaml", "r"))
logger = logging.getLogger("train")


class LLM_Scheduler:
    def __init__(self, agent_graph, action_graph):
        self.agent_graph = agent_graph
        self.action_graph = action_graph
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes

    def _build_messages(self, global_info, max_num, decision_mode):
        role_list = global_info.agent_role_list()
        history = self.agent_graph.get_agent_dialog_history(
            role_list,
            question=global_info.task.get("Question"),
        )
        candidates = [
            {
                "index": idx,
                "name": role,
                "hash": self.agent_hash_list[idx],
            }
            for idx, role in enumerate(self.agent_role_list)
        ]
        system_prompt = {
            "role": "system",
            "content": (
                "You are the scheduler for a research-oriented multi-agent system. "
                "Your job is to decide which specialist agents should act next for complex question answering, "
                "information retrieval, web search, evidence collection, analysis, and final response synthesis. "
                "Use the task, current workflow state, and dialogue history to choose the next agent or agents. "
                "Prefer agents that reduce uncertainty, fetch missing evidence, or synthesize when enough evidence exists. "
                "Avoid redundant calls unless confidence remains low or the task clearly benefits from parallel exploration. "
                f"Decision mode: {decision_mode}. "
                f"You may select up to {max_num} agents. "
                "Return strict JSON only. "
                'For multi-agent mode, use: {"selected_agents": [{"name": "agent_name", "confidence": 0.72}]}. '
                'For single-agent mode, use: {"selected_agent": {"name": "agent_name", "confidence": 0.91}}. '
                "Confidence must be in [0, 1]. Use agent names exactly as provided. Do not output any extra text."
            ),
        }
        user_prompt = {
            "role": "user",
            "content": (
                f"Task: {global_info.task.get('Question', '')}\n"
                f"Current workflow state: {getattr(global_info.workflow, 'state', 'unknown')}\n"
                "Scheduling objective: choose the next best agent execution plan for the current step.\n"
                f"Available agents: {json.dumps(candidates, ensure_ascii=True)}"
            ),
        }
        return [system_prompt, user_prompt] + history

    def _parse_json_block(self, response_text):
        fenced_match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", response_text, re.IGNORECASE)
        candidates = [response_text.strip()]
        if fenced_match:
            candidates.insert(0, fenced_match.group(1))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _normalize_confidence(self, confidence, default_value):
        try:
            confidence = float(confidence)
        except Exception:
            confidence = default_value
        return max(0.0, min(1.0, confidence))

    def _fallback_decision(self, response_text, decision_mode):
        selected_roles = []
        normalized_text = response_text.lower()
        for role in self.agent_role_list:
            if role.lower() in normalized_text:
                selected_roles.append(role)

        if not selected_roles:
            return []

        if decision_mode == "single_best":
            selected_roles = selected_roles[:1]

        confidence = 1.0 / max(1, len(selected_roles))
        return [{"name": role, "confidence": confidence} for role in selected_roles]

    def _extract_selected_agents(self, response_text, decision_mode):
        parsed = self._parse_json_block(response_text)
        if parsed is not None:
            if decision_mode == "single_best" and isinstance(parsed.get("selected_agent"), dict):
                agent = parsed["selected_agent"]
                name = str(agent.get("name", "")).strip()
                if name:
                    return [
                        {
                            "name": name,
                            "confidence": self._normalize_confidence(agent.get("confidence", 1.0), 1.0),
                        }
                    ]

            selected_agents = parsed.get("selected_agents", [])
            if isinstance(selected_agents, list):
                normalized = []
                for item in selected_agents:
                    if isinstance(item, dict):
                        name = str(item.get("name", "")).strip()
                        if not name:
                            continue
                        normalized.append(
                            {
                                "name": name,
                                "confidence": self._normalize_confidence(item.get("confidence", 0.5), 0.5),
                            }
                        )
                    elif isinstance(item, str) and item.strip():
                        normalized.append({"name": item.strip(), "confidence": 0.5})
                if decision_mode == "single_best":
                    return normalized[:1]
                return normalized

        return self._fallback_decision(response_text, decision_mode)

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def forward(self, global_info, max_num=1, decision_mode="multi"):
        messages = self._build_messages(global_info, max_num, decision_mode)
        response_text, _ = query_gpt(messages)
        selected_agents = self._extract_selected_agents(response_text, decision_mode)

        decisions = []
        seen_roles = set()
        for agent in selected_agents:
            role = agent["name"]
            if role not in self.agent_role_list or role in seen_roles:
                continue
            role_idx = self.agent_role_list.index(role)
            decisions.append(
                {
                    "name": role,
                    "hash": self.agent_hash_list[role_idx],
                    "index": role_idx,
                    "confidence": agent["confidence"],
                }
            )
            seen_roles.add(role)

        if not decisions:
            raise ValueError(f"Failed to parse selected agents from LLM response: {response_text}")

        decisions = sorted(decisions, key=lambda item: item["confidence"], reverse=True)
        if decision_mode == "single_best":
            return decisions[:1]
        return decisions[:max_num]


@Singleton
class ContinuousREINFORCE(LearningPolicy):
    def __init__(self, agent_graph, action_graph, config_path="config/policy-old.json"):
        super().__init__(agent_graph, action_graph)
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.device = self.config["device"]["type"]
        self.model_path = self.config["paths"]["model_path"]
        self.training = self.config["training"]["training"]
        self.loading = self.config["training"]["loading"]
        self.learning_rate = self.config["training"]["learning_rate"]
        self.gamma = self.config["training"]["gamma"]
        self.sample_size = self.config["training"]["sample_size"]
        self.lambda_kl_loss = self.config["training"]["lambda_kl_loss"]

        self.max_num_agents = self.config["agent"]["max_num_agents"]
        self.next_num_agents = self.config["agent"]["next_num_agents"]
        self.max_path = self.config["agent"]["max_path"]
        self.threshold = self.config["agent"]["threshold"]

        self.llm_prior = self.config["llm"]["prior"]
        self.llm_prior_redistribution = self.config["llm"]["prior_redistribution"]
        self.redistribution_weight = self.config["llm"]["redistribution_weight"]
        scheduler_config = self.config.get("llm_scheduler", {})
        self.scheduler_mode = scheduler_config.get("mode", "multi")
        self.scheduler_default_top_k = scheduler_config.get("top_k", min(self.max_num_agents, self.max_path))

        self.llm_scheduler = LLM_Scheduler(self.agent_graph, self.action_graph)
        self.optimizer = None

        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes

        self.executed_trajectories = []
        self.execution_count = 0
        self.current_trajectories = []
        self.current_trajectory_idx = 0

        self.policy_losses = []
        self.rewards_history = []
        self.action_probs_history = []
        self.llm_action_probs_history = []
        self.reward_from_rm = []
        self.accumulated_acc = []
        self.entropy_history = []

        self.end_action = torch.tensor(self.agent_graph.terminator_agent_index, device=self.device)
        self.web_actions = torch.tensor(self.agent_graph.search_agent_indices, device=self.device)

        reward_factors = self.config["agent"]["reward_factors"]
        self.agent_reward_factor = [reward_factors["default"]] * self.actions_dim
        self.agent_reward_factor[self.end_action.item()] = reward_factors["terminator"]
        for web_idx in self.web_actions:
            self.agent_reward_factor[web_idx.item()] = reward_factors["web_search"]

        self.current_task = None
        self.previous_task = None
        self.global_step = 0
        self.prob_step = 0
        self.max_step_num = global_config.get("graph", {}).get("max_step_num", 1)

    def logarithmic_cost(self, step):
        scale = self.config["cost"]["scale"]
        growth_rate = self.config["cost"]["growth_rate"]
        normalized_step = (step + 1) / (self.max_step_num + 1)

        if self.config["cost"]["inverse"]:
            step_cost = scale * (
                1
                - torch.log(torch.tensor(1 + growth_rate * normalized_step, device=self.device))
                / torch.log(torch.tensor(1 + growth_rate, device=self.device))
            )
        else:
            step_cost = scale * (
                torch.log(torch.tensor(1 + growth_rate * normalized_step, device=self.device))
                / torch.log(torch.tensor(1 + growth_rate, device=self.device))
            )
        print("\033[1;33mstep cost: {}\033[0m".format(step_cost))
        return step_cost

    def save_model(self, path=None, tag=None):
        path = path or self.config["paths"]["checkpoint_path"]
        os.makedirs(path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"policy_net_{timestamp}" + (f"_{tag}" if tag else "") + ".pt"
        save_path = os.path.join(path, filename)

        checkpoint = {
            "timestamp": timestamp,
            "config": self.config,
            "metadata": {
                "tag": tag,
                "version": "llm-only",
            },
        }

        try:
            torch.save(checkpoint, save_path)
            print(f"Model saved successfully to {save_path}")
            return save_path
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None

    def load_model(self, path, strict=True):
        try:
            if not path or not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False

            checkpoint = torch.load(path, map_location="cpu")
            if "config" in checkpoint and isinstance(checkpoint["config"], dict):
                self.config.update({k: v for k, v in checkpoint["config"].items() if k not in self.config})

            print(f"Model loaded successfully from {path}")
            print(f"Model timestamp: {checkpoint.get('timestamp')}")
            if checkpoint.get("metadata", {}).get("tag"):
                print(f"Model tag: {checkpoint['metadata']['tag']}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def update_executed_trajectories(self):
        if self.current_task != self.previous_task:
            self.previous_task = self.current_task
            self.execution_count += 1
            num_to_add = self.execution_count - len(self.executed_trajectories)
            if num_to_add > 0:
                self.executed_trajectories.extend([[] for _ in range(num_to_add)])
        self.current_trajectories = self.executed_trajectories[self.execution_count - 1]

    def _record_action_distribution(self, decisions):
        action_probs = torch.zeros(self.actions_dim, device=self.device)
        if len(decisions) == 0:
            action_probs[self.end_action.item()] = 1.0
        else:
            confidences = torch.tensor(
                [max(float(item["confidence"]), 1e-6) for item in decisions],
                device=self.device,
                dtype=torch.float32,
            )
            confidences = confidences / confidences.sum()
            for item, prob in zip(decisions, confidences):
                action_probs[item["index"]] = prob

        self.action_probs_history.append(action_probs)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
        self.entropy_history.append(entropy)
        return action_probs

    def _decide_agent_indices(self, global_info):
        if self.scheduler_mode == "single_best":
            max_num = 1
        else:
            max_num = max(1, min(self.scheduler_default_top_k, self.max_num_agents, self.max_path))
        try:
            decisions = self.llm_scheduler.forward(global_info, max_num=max_num, decision_mode=self.scheduler_mode)
        except Exception as exc:
            logger.warning(f"LLM policy failed: {exc}. Defaulting to terminator.")
            fallback = [{"name": self.agent_role_list[self.end_action.item()], "index": self.end_action.item(), "confidence": 1.0}]
            return torch.tensor([self.end_action.item()], device=self.device), fallback

        selected_indices = [item["index"] for item in decisions]
        if not selected_indices:
            logger.warning("LLM policy returned no valid agents. Defaulting to terminator.")
            decisions = [{"name": self.agent_role_list[self.end_action.item()], "index": self.end_action.item(), "confidence": 1.0}]
            selected_indices = [self.end_action.item()]

        deduped_indices = list(dict.fromkeys(selected_indices))[:max_num]
        deduped_decisions = []
        seen_indices = set()
        for item in decisions:
            if item["index"] in seen_indices or item["index"] not in deduped_indices:
                continue
            deduped_decisions.append(item)
            seen_indices.add(item["index"])
        return torch.tensor(deduped_indices, device=self.device), deduped_decisions

    def forward(self, global_info):
        if global_info.path_id == -1:
            agent_indices = self.init_forward(global_info)
        else:
            agent_indices = self.iter_forward(global_info)
        print("Agent Indices: {}".format(agent_indices))
        return [self.agent_hash_list[i] for i in agent_indices]

    def init_forward(self, global_info):
        print("\033[1;33m[LLM Only] Init Policy Forward\033[0m")
        logger.info("[Init Policy Forward]")
        self.current_task = global_info.task
        self.update_executed_trajectories()

        agent_indices, decisions = self._decide_agent_indices(global_info)
        action_probs = self._record_action_distribution(decisions)

        self.current_trajectory_idx = 0
        length = len(self.current_trajectories) + agent_indices.shape[0]
        while len(self.current_trajectories) < length:
            self.current_trajectories.append([])

        for i, agent_idx in enumerate(agent_indices):
            prob_value = action_probs[agent_idx.item()]
            if i == 0:
                trajectory_idx = self.current_trajectory_idx
            else:
                trajectory_idx = len(self.current_trajectories) - len(agent_indices) + i
            self.append_to_trajectory(trajectory_idx, agent_idx, prob_value, global_info, None, None, 0)

        return agent_indices

    def iter_forward(self, global_info):
        print("\033[1;33m[LLM Only] Following Policy Forward\033[0m")
        logger.info("Following Policy Forward")

        self.current_task = global_info.task
        agent_indices, decisions = self._decide_agent_indices(global_info)
        action_probs = self._record_action_distribution(decisions)

        self.current_trajectory_idx = global_info.path_id
        length = len(self.current_trajectories) + len(agent_indices) - 1
        original_length = len(self.current_trajectories)
        while len(self.current_trajectories) < length:
            self.current_trajectories.append([])

        for i, agent_idx in enumerate(agent_indices):
            prob_value = action_probs[agent_idx.item()]
            if i == 0:
                trajectory_idx = self.current_trajectory_idx
            else:
                trajectory_idx = original_length + i - 1
                self.current_trajectories[trajectory_idx] = self.clone_trajectory(self.current_trajectory_idx)
            self.append_to_trajectory(trajectory_idx, agent_idx, prob_value, global_info, None, None, 0)

        return agent_indices

    def append_to_trajectory(self, trajectory_idx, agent_idx, prob_value, global_info, prior_action_probs, m, rew=0):
        log_prob_val = torch.log(prob_value + 1e-10)
        cost = self.logarithmic_cost(len(self.current_trajectories[trajectory_idx])) * self.agent_reward_factor[
            agent_idx.item()
        ]
        self.current_trajectories[trajectory_idx].append(
            {
                "prob": prob_value,
                "log_prob": log_prob_val,
                "state_identifier": global_info.workflow.state,
                "action": self.agent_role_list[agent_idx.item()],
                "reward": cost,
                "reward_model": rew,
                "prior_prob": None,
            }
        )
        print(trajectory_idx, self.current_trajectories[trajectory_idx])

    def clone_trajectory(self, source_idx):
        return [
            {
                "prob": t["prob"].clone(),
                "log_prob": t["log_prob"].clone(),
                "state_identifier": t["state_identifier"],
                "action": t["action"],
                "reward": t["reward"],
                "reward_model": t["reward_model"],
                "prior_prob": t["prior_prob"].clone() if t["prior_prob"] is not None else None,
            }
            for t in self.current_trajectories[source_idx][:-1]
        ]

    def calculate_returns(self, trajectory):
        returns = []
        R = 0
        for t in reversed(trajectory):
            R = t.get("reward", 0) + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)

    def update(self):
        print("Inference mode: Skipping Update")
        self.current_trajectories = []
        self.executed_trajectories = []
        self.execution_count = 0
        return {}

    def finalize_task(self, transition, global_info):
        print("\033[1;33mtransition reward: {}\033[0m".format(transition.get("reward", 0)))
        if self.execution_count <= 0:
            self.rewards_history.append(transition.get("reward", 0))
            return

        self.current_trajectories = self.executed_trajectories[self.execution_count - 1]
        idx = transition.get("path_id", 0)
        if self.current_trajectories and idx < len(self.current_trajectories):
            current_trajectory = self.current_trajectories[idx]
            for index, action in enumerate(global_info.workflow.workflow):
                if index >= len(current_trajectory):
                    break
                cost = action.cost
                print("\033[1;33mtoken cost: {}\033[0m".format(cost))
                print("\033[1;33mcost factor: {}\033[0m".format(cost / 100000))
                current_trajectory[index]["reward"] *= cost / 100000
                print("\033[1;33mReward: {}\033[0m".format(current_trajectory[index]["reward"]))

            if current_trajectory:
                step_reward = self.logarithmic_cost(len(current_trajectory))
                total_tokens = global_info.total_tokens
                total_cost = global_info.total_cost
                if transition.get("reward", 0) > 0:
                    reward = transition.get("reward", 0) + self.agent_reward_factor[self.end_action.item()] * step_reward
                else:
                    reward = transition.get("reward", 0) - self.agent_reward_factor[self.end_action.item()] * step_reward

                if current_trajectory[-1].get("action") == self.agent_role_list[self.end_action.item()]:
                    current_trajectory[-1]["reward"] = reward
                    current_trajectory[-1]["total_tokens"] = total_tokens
                    current_trajectory[-1]["total_cost"] = total_cost
                    current_trajectory[-1]["finalized"] = True
                    current_trajectory[-1]["reward_model"] = 0
                    current_trajectory[-1]["metrics"] = transition.get("metrics", {})
                    print("\033[1;33mLast Reward: {}\033[0m".format(current_trajectory[-1]["reward"]))
                else:
                    current_trajectory.append(
                        {
                            "prob": torch.tensor(1.0, device=self.device),
                            "log_prob": torch.tensor(0.0, device=self.device),
                            "state_identifier": transition.get("state", global_info.workflow.state),
                            "action": self.agent_role_list[self.end_action.item()],
                            "reward": reward,
                            "reward_model": 0,
                            "finalized": True,
                            "total_tokens": total_tokens,
                            "total_cost": total_cost,
                            "metrics": transition.get("metrics", {}),
                        }
                    )
                    print("\033[1;33mLast Reward: {}\033[0m".format(current_trajectory[-1]["reward"]))
        self.rewards_history.append(transition.get("reward", 0))

    def select_agents_by_probability(self, action_probs):
        num_agents_to_select = torch.randint(1, self.max_num_agents + 1, (1,)).item()
        selected_indices = torch.multinomial(action_probs, num_agents_to_select, replacement=False)
        return selected_indices

    def select_agents_by_threshold(self, action_probs, threshold=0.1):
        threshold = 2 / self.agent_graph.num
        selected_indices = torch.nonzero(action_probs[0] > threshold).squeeze(1)
        if len(selected_indices) == 0:
            num_to_select = min(self.max_path, self.max_num_agents)
            selected_indices = torch.multinomial(action_probs, num_to_select, replacement=False)
            return selected_indices
        probs = action_probs[0][selected_indices]
        sorted_idx = torch.argsort(probs, descending=True)
        selected_indices = selected_indices[sorted_idx]

        num_agents_to_select = min(len(selected_indices), self.max_path, self.max_num_agents)
        selected_indices = selected_indices[:num_agents_to_select]
        return selected_indices.unsqueeze(0)

    def get_latest_model_path(self):
        try:
            path = self.model_path
            if os.path.exists(path) and os.path.isfile(path):
                return path

            path = self.config["paths"]["checkpoint_path"]
            if not os.path.exists(path):
                return None

            model_files = [f for f in os.listdir(path) if f.endswith(".pt")]
            if not model_files:
                return None

            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(path, x)))
            return os.path.join(path, latest_model)
        except Exception as e:
            print(f"Error finding latest model: {str(e)}")
            return None
