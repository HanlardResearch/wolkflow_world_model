from __future__ import annotations

import hashlib
import json
import os
import datetime
from typing import Any, Dict, List, Optional, Sequence

from agent.agent_info.actions import REASONING_ACTION_LIST

# 这个 recorder 的职责是把现有 scheduler / workflow 执行过程，
# 转换成“世界模型可训练”的 JSONL 样本。
#
# 关键思想：
# - 调度时先记录 scheduler 看到的 s_t
# - 任务完成后再用真实 workflow 结果把 s_{t+1}、reward、cost、done 回填回来
# - 最终形成 step-level 的 transition 样本
#
# 这样做的好处：
# - 不需要大改 agent 内部执行逻辑
# - 只在 policy 层就能拿到足够训练 world model 的监督数据
# - 输出 schema 和 WorkflowStateAdapter 直接对齐
#
# 输出路径说明：
# - 如果配置了 `world_model_dataset.output_dir`，则统一写到该目录
# - 如果 `use_dataset_subdirs=true`，则会继续在其下追加 dataset_name/dataset_mode 子目录
# - 如果 `split_by_time=true`，则文件名会自动追加时间后缀
# - 否则回退到当前任务的 `workpath`
# 这样既保留了原有行为，也支持通过超参数集中管理数据集路径与切分策略
#
# ----------------------------------------------------------------------------
# 推荐你把这个 recorder 输出理解成一个“离线环境回放数据集”：
# ----------------------------------------------------------------------------
# 每一行都近似对应：
#   (s_t, a_t, s_{t+1}, r_t, info_t)
#
# 其中：
# - s_t     来自 capture_decision_state()
# - a_t     来自 scheduler 最终选中的 agent/action
# - s_{t+1} 来自 finalize_task 时根据真实 workflow 前缀重建
# - r_t     来自 trajectory 中最终整理出的 reward
# - info_t  包括 token/cost/credit/辅助目标等
#
# 这也是为什么 recorder 要放在 policy 层而不是单个 agent 层：
# - policy 层能同时看到“决策前状态”和“最终路径归因”
# - agent 层通常只能看到自己的局部执行过程


class WorkflowDatasetRecorder:
    def __init__(
        self,
        agent_graph,
        recorder_config: Optional[Dict[str, Any]] = None,
        scheduler=None,
        max_step_num: int = 1,
    ) -> None:
        self.agent_graph = agent_graph
        self.scheduler = scheduler
        self.recorder_config = recorder_config or {}
        self.enabled = bool(self.recorder_config.get("enabled", True))
        self.filename = self.recorder_config.get("filename", "workflow_world_model.jsonl")
        self.output_dir = self.recorder_config.get("output_dir")
        self.use_dataset_subdirs = bool(self.recorder_config.get("use_dataset_subdirs", True))
        self.split_by_time = bool(self.recorder_config.get("split_by_time", False))
        self.time_granularity = str(self.recorder_config.get("time_granularity", "day")).lower()
        self.dataset_name = self.recorder_config.get("dataset_name")
        self.dataset_mode = self.recorder_config.get("dataset_mode")
        self.max_summary_chars = int(self.recorder_config.get("max_summary_chars", 240))
        self.include_synthetic_termination = bool(
            self.recorder_config.get("include_synthetic_termination", True)
        )
        self.max_step_num = max(int(max_step_num), 1)
        self.agent_role_list = list(getattr(agent_graph, "role_nodes", []))

    def capture_decision_state(self, global_info) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        workflow_actions = list(getattr(getattr(global_info, "workflow", None), "workflow", []))
        # 这里记录的是“动作执行前”的状态，也就是标准 MDP 里的 s_t。
        # 之所以在这里采，而不是在 finalize_task 里事后恢复，
        # 是因为 scheduler 在这一刻能看到的 state 才是决策依据。
        return self._build_state_snapshot(
            task=getattr(global_info, "task", {}),
            actions=workflow_actions,
            answers=list(getattr(global_info, "answers", [])),
            path_id=getattr(global_info, "path_id", -1),
            total_tokens=getattr(global_info, "total_tokens", 0),
            total_cost=getattr(global_info, "total_cost", 0),
        )

    def record_completed_trajectory(
        self,
        trajectory: Sequence[Dict[str, Any]],
        global_info,
        transition: Dict[str, Any],
        path_id: int,
        gamma: float,
    ) -> List[Dict[str, Any]]:
        # 这个函数在任务结束时调用。
        #
        # 输入：
        # - trajectory: policy 层记录的抽象轨迹，每步都带 pre-state snapshot
        # - global_info.workflow.workflow: 真实执行过的 action 列表
        #
        # 输出：
        # - 面向 world model 的 step-level records
        #
        # 它做的事情，本质上是把“抽象决策日志”和“真实执行结果”拼起来。
        if not self.enabled or not trajectory:
            return []
        if trajectory[-1].get("world_model_logged", False):
            return []

        workflow_actions = list(getattr(getattr(global_info, "workflow", None), "workflow", []))
        returns = self._calculate_returns(trajectory, gamma)
        episode_id = self._build_episode_id(global_info, path_id)
        metrics = dict(transition.get("metrics", {}))
        records: List[Dict[str, Any]] = []

        actual_step_count = min(len(workflow_actions), len(trajectory))
        has_synthetic_termination = (
            len(trajectory) > len(workflow_actions)
            and str(trajectory[-1].get("action", "")) == "TerminatorAgent"
        )

        for step_index in range(actual_step_count):
            # 逐步构建 transition：
            # state = 动作前 snapshot
            # action = 当前真实执行的 agent/action
            # next_state = 在真实 workflow 前缀基础上重建
            # outcome/return/credit = 从真实 reward 与轨迹回报中回填
            trajectory_step = trajectory[step_index]
            action = workflow_actions[step_index]
            pre_actions = workflow_actions[:step_index]
            prefix_actions = workflow_actions[: step_index + 1]
            pre_graph = self._build_graph_snapshot(actions=pre_actions, trajectory=trajectory[:step_index])
            post_graph = self._build_graph_snapshot(actions=prefix_actions, trajectory=trajectory[: step_index + 1])
            state_snapshot = trajectory_step.get("state_snapshot")
            if state_snapshot is None:
                state_snapshot = self._build_state_snapshot(
                    task=getattr(global_info, "task", {}),
                    actions=pre_actions,
                    answers=None,
                    path_id=path_id,
                    total_tokens=sum(item.tokens for item in pre_actions),
                    total_cost=sum(item.cost for item in pre_actions),
                    graph_snapshot=pre_graph,
                )
            next_state = self._build_state_snapshot(
                task=getattr(global_info, "task", {}),
                actions=prefix_actions,
                answers=None,
                path_id=path_id,
                total_tokens=sum(item.tokens for item in prefix_actions),
                total_cost=sum(item.cost for item in prefix_actions),
                graph_snapshot=post_graph,
            )
            done = step_index == actual_step_count - 1 and not has_synthetic_termination
            record = {
                "episode_id": episode_id,
                "path_id": path_id,
                "t": step_index,
                "task": dict(getattr(global_info, "task", {})),
                "graph": pre_graph,
                "state": state_snapshot.get("state", {}),
                "next_graph": post_graph,
                "next_state": next_state.get("state", {}),
                "action": {
                    "kind": "primitive",
                    "name": str(trajectory_step.get("action", action.agent_role)),
                    "selected_confidence": self._to_float(
                        trajectory_step.get("selected_confidence", trajectory_step.get("prob", 0.0))
                    ),
                    "estimated_cost": self._to_float(action.cost),
                    "candidate_agents": list(trajectory_step.get("candidate_agents", [])),
                },
                "next_state_targets": self._build_next_state_targets(
                    prefix_actions,
                    done=done,
                ),
                "outcome": {
                    "reward": self._to_float(trajectory_step.get("reward", 0.0)),
                    "cost_delta": self._to_float(action.cost),
                    "token_delta": self._to_float(action.tokens),
                    "done": done,
                    "success": self._normalize_success(action.success),
                },
                "returns": {
                    "mc_return": returns[step_index],
                    "h2_return": self._discounted_window_return(trajectory, step_index, horizon=2, gamma=gamma),
                },
                "credit_targets": {
                    "leave_one_out_gap": returns[step_index],
                    "step_credit": self._to_float(trajectory_step.get("reward", 0.0)),
                },
                "metadata": {
                    "agent_role": action.agent_role,
                    "action_name": str(action.action.get("action")),
                    "action_parameter": self._summarize_text(action.action.get("parameter"), limit=120),
                    "result_summary": self._summarize_text(action.result.get("step_data"), limit=240),
                    "answer_summary": self._summarize_text(action.result.get("answer"), limit=160),
                    "metrics": metrics,
                },
            }
            records.append(record)

        if self.include_synthetic_termination and has_synthetic_termination:
            # 如果 trajectory 末尾是 policy 人工补的 Terminator，
            # 这里额外补一个终止 step，方便世界模型显式学习 done transition。
            terminal_step = trajectory[-1]
            terminal_index = len(records)
            final_graph = self._build_graph_snapshot(actions=workflow_actions, trajectory=trajectory[:actual_step_count])
            final_state = self._build_state_snapshot(
                task=getattr(global_info, "task", {}),
                actions=workflow_actions,
                answers=None,
                path_id=path_id,
                total_tokens=getattr(global_info, "total_tokens", 0),
                total_cost=getattr(global_info, "total_cost", 0),
                graph_snapshot=final_graph,
            )
            records.append(
                {
                    "episode_id": episode_id,
                    "path_id": path_id,
                    "t": terminal_index,
                    "task": dict(getattr(global_info, "task", {})),
                    "graph": final_graph,
                    "state": final_state.get("state", {}),
                    "next_graph": final_graph,
                    "next_state": final_state.get("state", {}),
                    "action": {
                        "kind": "primitive",
                        "name": "TerminatorAgent",
                        "selected_confidence": self._to_float(
                            terminal_step.get("selected_confidence", terminal_step.get("prob", 1.0))
                        ),
                        "estimated_cost": 0.0,
                        "candidate_agents": ["TerminatorAgent"],
                    },
                    "next_state_targets": self._build_next_state_targets(workflow_actions, done=True),
                    "outcome": {
                        "reward": self._to_float(terminal_step.get("reward", 0.0)),
                        "cost_delta": 0.0,
                        "token_delta": 0.0,
                        "done": True,
                        "success": True,
                    },
                    "returns": {
                        "mc_return": returns[-1] if returns else self._to_float(terminal_step.get("reward", 0.0)),
                        "h2_return": returns[-1] if returns else self._to_float(terminal_step.get("reward", 0.0)),
                    },
                    "credit_targets": {
                        "leave_one_out_gap": returns[-1] if returns else self._to_float(terminal_step.get("reward", 0.0)),
                        "step_credit": self._to_float(terminal_step.get("reward", 0.0)),
                    },
                    "metadata": {"agent_role": "TerminatorAgent", "metrics": metrics},
                }
            )

        self._write_records(records, global_info)
        for step in trajectory:
            step["world_model_logged"] = True
        return records

    def _calculate_returns(self, trajectory: Sequence[Dict[str, Any]], gamma: float) -> List[float]:
        # 标准 discounted return。
        # 这里是 recorder 侧的后处理，用于给离线训练提供 Monte Carlo 目标。
        returns: List[float] = []
        running_return = 0.0
        for step in reversed(trajectory):
            running_return = self._to_float(step.get("reward", 0.0)) + gamma * running_return
            returns.insert(0, running_return)
        return returns

    def _discounted_window_return(
        self,
        trajectory: Sequence[Dict[str, Any]],
        start_index: int,
        horizon: int,
        gamma: float,
    ) -> float:
        # 一个短视距 return target，适合后续 world model / planner 做局部估值。
        total = 0.0
        for offset in range(horizon):
            index = start_index + offset
            if index >= len(trajectory):
                break
            total += (gamma**offset) * self._to_float(trajectory[index].get("reward", 0.0))
        return total

    def _build_state_snapshot(
        self,
        task: Dict[str, Any],
        actions: Sequence[Any],
        answers: Optional[Sequence[Any]],
        path_id: int,
        total_tokens: Any,
        total_cost: Any,
        graph_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 这个函数负责把运行时对象压缩成一个静态、可序列化、可训练的状态表示。
        #
        # 设计要求：
        # 1. 尽量贴合当前 scheduler 可见的信息
        # 2. 只保留稳定字段，避免把复杂对象直接写入 JSON
        # 3. 输出字段名与 WorkflowStateAdapter 对齐
        workflow_summary = self._build_workflow_summary(actions, answers)
        return {
            "graph": graph_snapshot if graph_snapshot is not None else self._build_graph_snapshot(actions=actions),
            "state": {
                "workflow_state": workflow_summary["state"],
                "executed_steps": workflow_summary["executed_steps"],
                "recent_answers": workflow_summary["recent_answers"],
                "reasoning_results": workflow_summary["reasoning_results"],
                "tool_results": workflow_summary["tool_results"],
                "all_actions": workflow_summary["all_actions"],
                "valid_actions": list(self.agent_role_list),
                "budget": {
                    "step_index": len(actions),
                    "used_tokens": self._to_float(total_tokens),
                    "used_cost": self._to_float(total_cost),
                },
                "workflow_valid_actions": workflow_summary["workflow_valid_actions"],
                "path_id": path_id,
            },
            "task": dict(task),
        }

    def _build_workflow_summary(
        self,
        actions: Sequence[Any],
        answers: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        # 这部分逻辑基本对应 LLM scheduler 看到的 workflow summary，
        # 只是这里改成纯静态函数，方便 recorder 在 finalize 时重建任意前缀状态。
        executed_steps: List[Dict[str, Any]] = []
        reasoning_results: List[str] = []
        tool_results: List[str] = []
        all_actions: List[str] = []
        workflow_valid_actions: List[str] = []

        for step_index, action in enumerate(actions):
            # 每个 executed step 只保留对调度有用的核心信息：
            # agent / action / success / summary / tokens / cost
            action_name = str(action.action.get("action"))
            success = self._normalize_success(action.success)
            step_data = self._summarize_text(action.result.get("step_data"), limit=300)
            answer_summary = self._summarize_text(action.result.get("answer"), limit=160)
            executed_steps.append(
                {
                    "step_index": step_index,
                    "agent": action.agent_role,
                    "action": action_name,
                    "parameter": self._summarize_text(action.action.get("parameter"), limit=120),
                    "success": success,
                    "step_data_summary": step_data,
                    "answer_summary": answer_summary,
                    "tokens": self._to_float(action.tokens),
                    "cost": self._to_float(action.cost),
                }
            )
            all_actions.append(action_name)
            if success:
                workflow_valid_actions.append(action_name)
                result_line = f"Successful Action: {action_name}\nResult: {step_data}"
                if action_name in REASONING_ACTION_LIST:
                    reasoning_results.append(result_line)
                else:
                    tool_results.append(result_line)

        recent_answers: List[str] = []
        if answers:
            recent_answers = [self._summarize_text(item, limit=160) for item in answers if item is not None][-3:]
        elif actions:
            recent_answers = [
                self._summarize_text(action.result.get("answer"), limit=160)
                for action in actions
                if action.result.get("answer")
            ][-3:]

        state = tuple(
            (
                action.agent_role,
                action.action.get("action"),
                1 if self._normalize_success(action.success) else 0,
            )
            for action in actions
        )
        if not state:
            state = tuple([(None, None, -1)])

        return {
            "executed_steps": executed_steps,
            "recent_answers": recent_answers,
            "state": str(state),
            "all_actions": all_actions,
            "workflow_valid_actions": workflow_valid_actions,
            "reasoning_results": reasoning_results[-5:],
            "tool_results": tool_results[-5:],
        }

    def _build_next_state_targets(self, actions: Sequence[Any], done: bool) -> Dict[str, Any]:
        # 这些 scalar target 目前是启发式构造的 dense supervision。
        #
        # 目的不是追求绝对完美，而是先提供一个足够稳定的训练信号：
        # - progress: 当前推进程度
        # - coverage: 已覆盖的信息面
        # - conflict: 失败/冲突程度
        # - redundancy: 重复调用程度
        # - termination_readiness: 是否接近可以结束
        #
        # 后面如果你接 judge model 或更强 reward model，可以替换这里。
        total_steps = max(len(actions), 1)
        successful_steps = sum(1 for action in actions if self._normalize_success(action.success))
        failed_steps = len(actions) - successful_steps
        successful_action_names = [
            str(action.action.get("action"))
            for action in actions
            if self._normalize_success(action.success)
        ]
        unique_successes = len(set(successful_action_names))
        reasoning_successes = sum(
            1
            for action in actions
            if self._normalize_success(action.success) and str(action.action.get("action")) in REASONING_ACTION_LIST
        )
        tool_successes = successful_steps - reasoning_successes
        redundancy = max(len(successful_action_names) - unique_successes, 0) / total_steps
        conflict = failed_steps / total_steps
        progress = min(successful_steps / self.max_step_num, 1.0)
        coverage = min((reasoning_successes + tool_successes) / total_steps, 1.0)
        readiness = 1.0 if done else min(
            1.0,
            0.35 * (1.0 if reasoning_successes > 0 else 0.0)
            + 0.35 * (1.0 if tool_successes > 0 else 0.0)
            + 0.30 * (1.0 - conflict),
        )
        return {
            "progress_score": progress,
            "coverage_score": coverage,
            "conflict_score": conflict,
            "redundancy_score": redundancy,
            "termination_readiness": readiness,
            "valid_action_mask": list(self.agent_role_list),
        }

    def _build_graph_snapshot(
        self,
        actions: Optional[Sequence[Any]] = None,
        trajectory: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # 图快照包含两部分：
        # 1. 静态结构：nodes / edges
        # 2. 动态统计：usage / success_rate / avg_cost / avg_credit
        #
        # 第二部分尤其重要，因为它给世界模型提供了“图是活的”这一层信息。
        edges: List[List[str]] = []
        prefix_actions = list(actions or [])
        prefix_trajectory = list(trajectory or [])
        seen_edges = set()
        for previous_action, current_action in zip(prefix_actions[:-1], prefix_actions[1:]):
            source = str(getattr(previous_action, "agent_role", "") or "")
            target = str(getattr(current_action, "agent_role", "") or "")
            if not source or not target:
                continue
            edge = (source, target)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append([source, target])
        ordered_roles = list(self.agent_role_list)
        node_totals: Dict[str, Dict[str, float]] = {
            role: {
                "usage_count": 0.0,
                "success_count": 0.0,
                "total_cost": 0.0,
                "total_credit": 0.0,
                "total_reward": 0.0,
            }
            for role in ordered_roles
        }
        node_stats = {}
        for index, action in enumerate(prefix_actions):
            role = str(getattr(action, "agent_role", "") or "")
            if not role:
                continue
            if role not in node_totals:
                ordered_roles.append(role)
                node_totals[role] = {
                    "usage_count": 0.0,
                    "success_count": 0.0,
                    "total_cost": 0.0,
                    "total_credit": 0.0,
                    "total_reward": 0.0,
                }
            stats = node_totals[role]
            stats["usage_count"] += 1.0
            stats["success_count"] += float(self._normalize_success(getattr(action, "success", False)))
            stats["total_cost"] += self._to_float(getattr(action, "cost", 0.0))
            if index < len(prefix_trajectory):
                reward_value = self._to_float(prefix_trajectory[index].get("reward", 0.0))
                stats["total_credit"] += reward_value
                stats["total_reward"] += reward_value
        for role in ordered_roles:
            stats = node_totals[role]
            usage = max(int(stats["usage_count"]), 0)
            success_rate = float(stats["success_count"]) / usage if usage > 0 else 0.0
            node_stats[role] = {
                "success_rate": success_rate,
                "avg_cost": float(stats["total_cost"]) / usage if usage > 0 else 0.0,
                "avg_credit": float(stats["total_credit"]) / usage if usage > 0 else 0.0,
                "avg_reward": float(stats["total_reward"]) / usage if usage > 0 else 0.0,
                "usage_count": usage,
            }
        return {"nodes": ordered_roles, "edges": edges, "node_stats": node_stats}

    def _update_agent_stats(self, actions: Sequence[Any], trajectory: Sequence[Dict[str, Any]]) -> None:
        return None
        # 用真实执行数据持续更新节点统计。
        # 这些统计会在下一次 snapshot 时进入 graph.node_stats。
        for index, action in enumerate(actions):
            role = str(action.agent_role)
            if role not in self.agent_stats:
                self.agent_stats[role] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "total_cost": 0.0,
                    "total_credit": 0.0,
                    "total_reward": 0.0,
                }
            stats = self.agent_stats[role]
            stats["usage_count"] += 1
            stats["success_count"] += int(self._normalize_success(action.success))
            stats["total_cost"] += self._to_float(action.cost)
            if index < len(trajectory):
                stats["total_credit"] += self._to_float(trajectory[index].get("reward", 0.0))
                stats["total_reward"] += self._to_float(trajectory[index].get("reward", 0.0))

    def _write_records(self, records: Sequence[Dict[str, Any]], global_info) -> None:
        # 按 JSONL 逐行追加，方便长时间采集和后续增量训练。
        target_path = self._resolve_output_path(global_info)
        if not target_path or not records:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _resolve_output_path(self, global_info) -> Optional[str]:
        # 优先使用超参数里的 output_dir。
        # 这允许将 world-model 数据集中写到一个统一目录，方便离线训练。
        workpath = getattr(global_info, "workpath", None)
        if self.output_dir:
            output_dir = str(self.output_dir)
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)
        elif workpath:
            output_dir = workpath
        else:
            return None

        if self.use_dataset_subdirs:
            dataset_name = self._resolve_dataset_name(global_info)
            dataset_mode = self._resolve_dataset_mode(global_info)
            if dataset_name:
                output_dir = os.path.join(output_dir, dataset_name)
            if dataset_mode:
                output_dir = os.path.join(output_dir, dataset_mode)

        filename = self._resolve_filename()
        return os.path.join(output_dir, filename)

    def _resolve_filename(self) -> str:
        if not self.split_by_time:
            return self.filename
        stem, ext = os.path.splitext(self.filename)
        return f"{stem}_{self._time_suffix()}{ext}"

    def _resolve_dataset_name(self, global_info) -> str:
        if self.dataset_name:
            return self._sanitize_path_component(self.dataset_name)
        task = getattr(global_info, "task", {}) or {}
        dataset_name = task.get("dataset_name") or task.get("type")
        return self._sanitize_path_component(dataset_name)

    def _resolve_dataset_mode(self, global_info) -> str:
        if self.dataset_mode:
            return self._sanitize_path_component(self.dataset_mode)
        task = getattr(global_info, "task", {}) or {}
        dataset_mode = task.get("dataset_mode") or task.get("mode")
        return self._sanitize_path_component(dataset_mode)

    def _time_suffix(self) -> str:
        now = datetime.datetime.now()
        format_map = {
            "day": "%Y%m%d",
            "hour": "%Y%m%d_%H",
            "minute": "%Y%m%d_%H%M",
            "second": "%Y%m%d_%H%M%S",
        }
        time_format = format_map.get(self.time_granularity, format_map["day"])
        return now.strftime(time_format)

    def _sanitize_path_component(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
            text = text.replace(char, "_")
        return text

    def _build_episode_id(self, global_info, path_id: int) -> str:
        # 用 path_id + workpath + question 构造稳定 episode id，
        # 便于离线训练时按 episode 切分 train/val。
        task = getattr(global_info, "task", {}) or {}
        question = str(task.get("Question", task.get("question", "")))
        raw = f"{path_id}|{getattr(global_info, 'workpath', '')}|{question}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _summarize_text(self, value: Any, limit: Optional[int] = None) -> str:
        # 优先复用 scheduler 内已有的摘要逻辑，避免两个模块对文本截断策略不一致。
        limit = self.max_summary_chars if limit is None else limit
        if hasattr(self.scheduler, "_summarize_text"):
            return str(self.scheduler._summarize_text(value, limit=limit))
        text = str(value or "").replace("\n", " ").strip()
        return text if len(text) <= limit else text[:limit] + "..."

    def _normalize_success(self, success: Any) -> bool:
        # 兼容布尔值和字符串 "Success"/"Failure" 两种表示。
        if isinstance(success, bool):
            return success
        return str(success).strip().lower() == "success"

    def _to_float(self, value: Any) -> float:
        # 统一处理 tensor / 标量 / None，方便日志序列化。
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return 0.0
