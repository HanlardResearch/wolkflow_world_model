import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.policy.base_policy import LearningPolicy
from tenacity import RetryError
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.other_utils import Singleton

global_config = yaml.safe_load(open("./config/global.yaml", "r"))
logger = logging.getLogger("train")


class LLM_Scheduler:
    def __init__(self, agent_graph, action_graph, scheduler_config):
        self.agent_graph = agent_graph
        self.action_graph = action_graph
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes
        self.scheduler_config = scheduler_config or {}
        self.backend = self.scheduler_config.get("backend", "transformers").lower()
        self.model_name = self.scheduler_config.get("model_name", "Qwen3.5-4B")
        self.model_path = self.scheduler_config.get("model_path")
        self.generation_config = self.scheduler_config.get("generation", {})
        self.tensor_parallel_size = int(self.scheduler_config.get("tensor_parallel_size", 1))
        self.gpu_ids = self.scheduler_config.get("gpu_ids", [0])
        self.gpu_memory_utilization = float(self.scheduler_config.get("gpu_memory_utilization", 0.9))
        self.max_model_len = int(self.scheduler_config.get("max_model_len", 8192))
        self.dtype = self.scheduler_config.get("dtype", "bfloat16")
        self.enforce_eager = bool(self.scheduler_config.get("enforce_eager", False))
        self.force_deterministic_generation = bool(
            self.scheduler_config.get("force_deterministic_generation", True)
        )
        self._engine = None
        self._tokenizer = None
        self._scheduler_loggers = {}
        self.mode2description = {
            "multi": (
                "Select multiple useful agents for the next step when the task benefits from parallel exploration, "
                "cross-checking, or a decomposition into retrieval, reasoning, and synthesis."
            ),
            "single_best": (
                "Select exactly one best next agent for the current step. "
                "Use this when the workflow should stay focused and only the highest-value next action is needed."
            ),
        }
        self._init_engine()

    def _get_agent_metadata(self, role_name):
        metadata_map = {
            "FileAgent": {
                "role_summary": "Reads or inspects local files and extracts information from them.",
                "when_to_use": "Use when the task depends on local documents, saved artifacts, or file contents.",
                "when_not_to_use": "Do not use for pure reasoning when no file evidence is needed.",
                "cost_level": "medium",
                "needs_external_tools": False,
                "best_for_stage": "evidence",
                "output_type": "file_evidence",
            },
            "ArxivAgent": {
                "role_summary": "Searches arXiv-style academic sources for research papers and technical evidence.",
                "when_to_use": "Use when the task needs academic literature, papers, or research-grounded evidence.",
                "when_not_to_use": "Do not use for simple common-knowledge or direct reasoning tasks.",
                "cost_level": "high",
                "needs_external_tools": True,
                "best_for_stage": "evidence",
                "output_type": "paper_evidence",
            },
            "TavilyAgent": {
                "role_summary": "Performs broad web search to retrieve current or external information.",
                "when_to_use": "Use when the task requires web evidence, factual lookup, or external information gathering.",
                "when_not_to_use": "Do not use when the question can be solved from internal reasoning alone.",
                "cost_level": "high",
                "needs_external_tools": True,
                "best_for_stage": "evidence",
                "output_type": "web_evidence",
            },
            "WebsiteAgent": {
                "role_summary": "Opens and inspects specific websites or URLs to extract relevant details.",
                "when_to_use": "Use when a concrete website or URL should be read instead of broad search.",
                "when_not_to_use": "Do not use for purely local reasoning or when no website is involved.",
                "cost_level": "high",
                "needs_external_tools": True,
                "best_for_stage": "evidence",
                "output_type": "website_evidence",
            },
            "TerminatorAgent": {
                "role_summary": "Stops the workflow when enough information has been gathered.",
                "when_to_use": "Use when the task is already solved or no further agent call is necessary.",
                "when_not_to_use": "Do not use when key evidence or reasoning is still missing.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "final",
                "output_type": "termination",
            },
            "PythonAgent_gpt4o": {
                "role_summary": "Executes code-oriented reasoning, calculations, and programmatic verification.",
                "when_to_use": "Use for math, symbolic manipulation, algorithmic verification, or code-based checking.",
                "when_not_to_use": "Do not use when lightweight text reasoning is sufficient and no computation is needed.",
                "cost_level": "medium",
                "needs_external_tools": False,
                "best_for_stage": "reasoning",
                "output_type": "computed_result",
            },
            "PlannerAgent_gpt4o": {
                "role_summary": "Plans decomposition steps and proposes a structured solution strategy.",
                "when_to_use": "Use at the beginning of complex tasks that benefit from explicit decomposition.",
                "when_not_to_use": "Do not use for short direct questions that can be solved immediately.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "initial",
                "output_type": "plan",
            },
            "ReasoningAgent_gpt4o": {
                "role_summary": "Performs direct logical, mathematical, and textual reasoning.",
                "when_to_use": "Use for deriving answers from existing context without needing external search.",
                "when_not_to_use": "Do not use when the main bottleneck is missing evidence rather than reasoning.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "reasoning",
                "output_type": "reasoned_solution",
            },
            "CriticAgent_gpt4o": {
                "role_summary": "Critiques previous reasoning and checks for errors, gaps, or weak assumptions.",
                "when_to_use": "Use when an existing answer needs validation or challenge before finalizing.",
                "when_not_to_use": "Do not use as the first step when no prior reasoning exists.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "verification",
                "output_type": "critique",
            },
            "ReflectAgent_gpt4o": {
                "role_summary": "Reflects on prior attempts and proposes higher-level corrections or improvements.",
                "when_to_use": "Use after a failed or weak reasoning attempt to redirect the workflow.",
                "when_not_to_use": "Do not use before any meaningful attempt has been made.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "recovery",
                "output_type": "reflection",
            },
            "QuestionAgent_gpt4o": {
                "role_summary": "Generates a useful sub-question or intermediate question to advance the task.",
                "when_to_use": "Use when the task should be decomposed into missing sub-problems.",
                "when_not_to_use": "Do not use when the solution path is already clear.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "decomposition",
                "output_type": "subquestion",
            },
            "SummarizerAgent_gpt4o": {
                "role_summary": "Summarizes prior evidence or reasoning into concise intermediate conclusions.",
                "when_to_use": "Use when multiple results need compression before the next step.",
                "when_not_to_use": "Do not use when there is little or no prior content to summarize.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "synthesis",
                "output_type": "summary",
            },
            "ConcluderAgent_gpt4o": {
                "role_summary": "Produces the final conclusion or final answer from existing evidence.",
                "when_to_use": "Use near the end when enough evidence and reasoning already exist.",
                "when_not_to_use": "Do not use too early when the task is not yet solved.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "final",
                "output_type": "final_answer",
            },
            "Modifier_gpt4o": {
                "role_summary": "Revises or corrects previous reasoning and rewrites flawed outputs.",
                "when_to_use": "Use when a previous answer exists but needs targeted correction or repair.",
                "when_not_to_use": "Do not use when there is no prior draft or reasoning to modify.",
                "cost_level": "low",
                "needs_external_tools": False,
                "best_for_stage": "repair",
                "output_type": "revised_answer",
            },
        }

        default_metadata = {
            "role_summary": "General-purpose agent in the multi-agent workflow.",
            "when_to_use": "Use when its name and context suggest it is the most relevant specialist.",
            "when_not_to_use": "Do not use when another agent is more clearly aligned with the task.",
            "cost_level": "medium",
            "needs_external_tools": False,
            "best_for_stage": "general",
            "output_type": "general_result",
        }
        return metadata_map.get(role_name, default_metadata)

    def _init_engine(self):
        if not self.model_path:
            raise ValueError("LLM scheduler requires `llm_scheduler.model_path` in config.")

        if self.backend == "vllm":
            self._init_vllm_engine()
        elif self.backend == "transformers":
            self._init_transformers_engine()
        else:
            raise ValueError(f"Unsupported scheduler backend: {self.backend}")

    def _init_vllm_engine(self):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vLLM backend requested but `vllm` is not installed.") from exc

        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._engine = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
        )
        logger.info(
            f"Initialized scheduler backend=vllm model={self.model_name} "
            f"gpu_ids={self.gpu_ids} tensor_parallel_size={self.tensor_parallel_size}"
        )

    def _init_transformers_engine(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        if len(self.gpu_ids) > 1:
            max_memory = {gpu_id: f"{int(self.gpu_memory_utilization * 80)}GiB" for gpu_id in self.gpu_ids}
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory
        else:
            target_device = f"cuda:{self.gpu_ids[0]}" if self.gpu_ids else "cpu"
            model_kwargs["device_map"] = {"": target_device}

        self._engine = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._engine.eval()
        logger.info(
            f"Initialized scheduler backend=transformers model={self.model_name} "
            f"gpu_ids={self.gpu_ids} tensor_parallel_size={self.tensor_parallel_size}"
        )

    def _get_scheduler_out_logger(self, global_info):
        log_dir = getattr(global_info, "workpath", None)
        if not log_dir:
            return None

        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "scheduler_out.log")
        if log_file in self._scheduler_loggers:
            return self._scheduler_loggers[log_file]

        scheduler_logger = logging.getLogger(f"scheduler_out::{log_file}")
        scheduler_logger.setLevel(logging.INFO)
        scheduler_logger.propagate = False

        if not scheduler_logger.handlers:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter('[%(asctime)s %(levelname)s]\n%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
            )
            scheduler_logger.addHandler(handler)

        self._scheduler_loggers[log_file] = scheduler_logger
        return scheduler_logger

    def _log_scheduler_exchange(
        self,
        global_info,
        messages,
        response_text,
        max_num,
        decision_mode,
        selected_agents=None,
        parsed=None,
        vllm_hp=None,
    ):
        scheduler_logger = self._get_scheduler_out_logger(global_info)
        if scheduler_logger is None:
            return

        log_payload = {
            "path_id": getattr(global_info, "path_id", None),
            "decision_mode": decision_mode,
            "max_num": max_num,
            "backend": self.backend,
            "model_name": self.model_name,
            "messages": messages,
            "response_text": response_text,
            "parsed": parsed,
            "selected_agents": selected_agents,
            "vllm_hp": vllm_hp,
        }
        scheduler_logger.info(json.dumps(log_payload, ensure_ascii=False, indent=2, default=str))

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_parts = []
        for message in messages:
            prompt_parts.append(f"{message['role'].upper()}: {message['content']}")
        prompt_parts.append("ASSISTANT:")
        return "\n".join(prompt_parts)

    def _get_vllm_hp(self):
        do_sample = bool(self.generation_config.get("do_sample", False))
        if self.force_deterministic_generation:
            do_sample = False
        temperature = float(self.generation_config.get("temperature", 0.1))
        top_p = float(self.generation_config.get("top_p", 0.9))
        top_k = int(self.generation_config.get("top_k", 20))
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": int(self.generation_config.get("max_new_tokens", 8192)),
            "repetition_penalty": float(self.generation_config.get("repetition_penalty", 1.08)),
            "frequency_penalty": float(self.generation_config.get("frequency_penalty", 0.25)),
            "presence_penalty": float(self.generation_config.get("presence_penalty", 0.15)),
            "do_sample": do_sample,
        }

    def _generate_with_vllm(self, messages: List[Dict[str, str]]) -> str:
        from vllm import SamplingParams

        prompt = self._messages_to_prompt(messages)
        vllm_hp = self._get_vllm_hp()
        sampling_params = SamplingParams(
            temperature=vllm_hp["temperature"],
            top_p=vllm_hp["top_p"],
            top_k=vllm_hp["top_k"],
            max_tokens=vllm_hp["max_tokens"],
            repetition_penalty=vllm_hp["repetition_penalty"],
            frequency_penalty=vllm_hp["frequency_penalty"],
            presence_penalty=vllm_hp["presence_penalty"],
        )
        outputs = self._engine.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            raise ValueError("Scheduler vLLM returned empty output.")
        return outputs[0].outputs[0].text

    def _get_transformers_generation_kwargs(self):
        do_sample = bool(self.generation_config.get("do_sample", False))
        if self.force_deterministic_generation:
            do_sample = False
        generation_kwargs = {
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 8192)),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(self.generation_config.get("temperature", 0.1)),
                    "top_p": float(self.generation_config.get("top_p", 0.9)),
                    "top_k": int(self.generation_config.get("top_k", 20)),
                }
            )
        return generation_kwargs

    def _generate_with_transformers(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._messages_to_prompt(messages)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        )
        model_device = next(self._engine.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._engine.generate(**inputs, **self._get_transformers_generation_kwargs())

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _sanitize_messages(self, messages):
        sanitized = []
        pending_system_content = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                if pending_system_content:
                    pending_system_content += f"\n[CONTEXT_FROM_SYSTEM]\n{content}"
                else:
                    pending_system_content = f"[CONTEXT_FROM_SYSTEM]\n{content}"
                continue

            if role == "user" and pending_system_content:
                merged_content = f"{pending_system_content}\n\n[ORIGINAL_USER_MESSAGE]\n{content}"
                sanitized.append({"role": "user", "content": merged_content})
                pending_system_content = None
                continue

            if pending_system_content:
                sanitized.append({"role": "user", "content": pending_system_content})
                pending_system_content = None

            sanitized.append({"role": role, "content": content})

        if pending_system_content:
            sanitized.append({"role": "user", "content": pending_system_content})

        sanitized.append(
            {
                "role": "user",
                "content": (
                    "[SCHEDULER_REMINDER]\n"
                    "Your task is still agent scheduling only. "
                    "Do not answer the user's domain question directly. "
                    "Do not continue the delegated agent's task. "
                    "Only decide which agent or agents should act next and return strict JSON."
                ),
            }
        )

        return sanitized

    def _summarize_text(self, text, limit=400):
        if text is None:
            return ""
        text = str(text).strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _build_workflow_summary(self, global_info):
        workflow = getattr(global_info, "workflow", None)
        if workflow is None or not getattr(workflow, "workflow", None):
            return {
                "executed_steps": [],
                "recent_answers": list(getattr(global_info, "answers", []))[-3:],
                "state": str(getattr(global_info, "workflow", None).state if getattr(global_info, "workflow", None) else []),
                "all_actions": [],
                "valid_actions": [],
                "reasoning_results": [],
                "tool_results": [],
            }

        executed_steps = []
        for idx, action in enumerate(workflow.workflow):
            executed_steps.append(
                {
                    "step_index": idx,
                    "agent": action.agent_role,
                    "action": action.action.get("action"),
                    "parameter": self._summarize_text(action.action.get("parameter"), limit=120),
                    "success": action.success,
                    "step_data_summary": self._summarize_text(action.result.get("step_data"), limit=300),
                    "answer_summary": self._summarize_text(action.result.get("answer"), limit=120),
                    "tokens": action.tokens,
                    "cost": action.cost,
                }
            )

        return {
            "executed_steps": executed_steps,
            "recent_answers": [self._summarize_text(answer, limit=120) for answer in list(getattr(global_info, "answers", []))[-3:]],
            "state": str(workflow.state),
            "all_actions": workflow.all_actions,
            "valid_actions": workflow.valid_actions,
            "reasoning_results": [self._summarize_text(item, limit=240) for item in workflow.valid_reasoning_results[-5:]],
            "tool_results": [self._summarize_text(item, limit=240) for item in workflow.valid_tool_results[-5:]],
        }

    def _build_messages(self, global_info, max_num, decision_mode):
        multi_mode_instruction = (
            f"You may return between 1 and {max_num} distinct agents in multi-agent mode, "
            "ordered from highest to lowest confidence. "
            "Return backup agents only when they provide clear additional value. "
            "If one agent is clearly the best next step, prefer fewer agents and assign it a substantially higher confidence than the backups."
        )
        candidates = [
            {
                "index": idx,
                "name": role,
                **self._get_agent_metadata(role),
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
                f"Decision mode keyword: {decision_mode} ({self.mode2description.get(decision_mode, decision_mode)}). "
                f"{multi_mode_instruction if decision_mode == 'multi' else 'You must return exactly one agent in single-agent mode.'} "
                f"Current workflow state: {getattr(global_info.workflow, 'state', 'unknown')}. "
                "Scheduling objective: choose the next best agent execution plan for the current step. "
                f"Available agents: {json.dumps(candidates, ensure_ascii=True)}. "
                "Return strict JSON only. "
                'For multi-agent mode, use: {"selected_agents": [{"name": "agent_name1", "confidence": 0.88}, {"name": "agent_name2", "confidence": 0.23}]}. '
                'For single-agent mode, use: {"selected_agent": {"name": "agent_name", "confidence": 0.91}}. '
                "Confidence must be in [0, 1]. Use agent names exactly as provided. Do not output any extra text."
            ),
        }
        workflow_summary = self._build_workflow_summary(global_info)
        user_prompt = {
            "role": "user",
            "content": (
                "Scheduler state summary:\n"
                f"{json.dumps({'task_question': global_info.task.get('Question', ''), 'workflow_summary': workflow_summary}, ensure_ascii=False)}\n\n"
                "Use the summary above as state only. "
                "Do not answer the task directly. "
                "Avoid repeating agents that already produced enough useful information unless verification or correction is needed. "
                "Prefer the minimal non-redundant next-step schedule."
            ),
        }
        return [system_prompt, user_prompt]

    def _parse_json_block(self, response_text):
        fenced_match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", response_text, re.IGNORECASE)
        candidates = []
        if fenced_match:
            candidates.append(fenced_match.group(1))

        stripped_text = response_text.strip()
        candidates.append(stripped_text)

        # Try to recover the last JSON object in verbose outputs like:
        # reasoning text ... {"selected_agent": {...}}
        for start_idx in range(len(stripped_text) - 1, -1, -1):
            if stripped_text[start_idx] != "{":
                continue
            candidate = stripped_text[start_idx:]
            candidates.append(candidate)

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

    def _fallback_decision(self, response_text, decision_mode, max_num=1):
        selected_roles = []
        normalized_text = response_text.lower()
        for role in self.agent_role_list:
            match_position = normalized_text.find(role.lower())
            if match_position >= 0:
                selected_roles.append((match_position, role))

        selected_roles = [role for _, role in sorted(selected_roles, key=lambda item: item[0])]

        if not selected_roles:
            return []

        if decision_mode == "single_best":
            selected_roles = selected_roles[:1]
        else:
            selected_roles = selected_roles[:max_num]

        confidence = 1.0 / max(1, len(selected_roles))
        return [{"name": role, "confidence": confidence} for role in selected_roles]

    def _get_default_multi_roles(self):
        preferred_roles = [
            "ReasoningAgent_gpt4o",
            "PythonAgent_gpt4o",
            "CriticAgent_gpt4o",
            "PlannerAgent_gpt4o",
            "TavilyAgent",
            "ArxivAgent",
            "WebsiteAgent",
            "QuestionAgent_gpt4o",
            "SummarizerAgent_gpt4o",
            "ReflectAgent_gpt4o",
            "Modifier_gpt4o",
            "ConcluderAgent_gpt4o",
            "FileAgent",
            "TerminatorAgent",
        ]
        ordered_roles = []
        seen_roles = set()
        for role in preferred_roles + self.agent_role_list:
            if role in self.agent_role_list and role not in seen_roles:
                ordered_roles.append(role)
                seen_roles.add(role)
        return ordered_roles

    def _complete_multi_selection(self, selected_agents, response_text, max_num):
        completed = []
        seen_roles = set()

        def append_candidates(candidates):
            for candidate in candidates:
                name = str(candidate.get("name", "")).strip()
                if not name or name not in self.agent_role_list or name in seen_roles:
                    continue
                completed.append(
                    {
                        "name": name,
                        "confidence": self._normalize_confidence(candidate.get("confidence", 0.2), 0.2),
                    }
                )
                seen_roles.add(name)
                if len(completed) >= max_num:
                    return

        append_candidates(selected_agents)
        if len(completed) == 0:
            fallback_candidates = self._fallback_decision(response_text, "multi", max_num=max_num)
            append_candidates(fallback_candidates)

        if len(completed) == 0:
            for role in self._get_default_multi_roles()[:1]:
                completed.append({"name": role, "confidence": 1.0})

        completed = sorted(completed, key=lambda item: item["confidence"], reverse=True)
        return completed[:max_num]

    def _extract_selected_agents(self, response_text, decision_mode, max_num, parsed=None):
        if parsed is None:
            parsed = self._parse_json_block(response_text)
        normalized = []
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

            if decision_mode == "multi" and isinstance(parsed.get("selected_agent"), dict):
                agent = parsed["selected_agent"]
                name = str(agent.get("name", "")).strip()
                if name:
                    normalized.append(
                        {
                            "name": name,
                            "confidence": self._normalize_confidence(agent.get("confidence", 1.0), 1.0),
                        }
                    )

            selected_agents = parsed.get("selected_agents", [])
            if isinstance(selected_agents, list):
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
                return self._complete_multi_selection(normalized, response_text, max_num)

        if decision_mode == "single_best":
            logger.warning("Scheduler JSON parsing failed, falling back to single-agent role-name matching.")
            return self._fallback_decision(response_text, decision_mode, max_num=1)

        logger.warning("Scheduler JSON parsing failed, falling back to multi-agent role-name matching.")
        return self._complete_multi_selection(normalized, response_text, max_num)

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def forward(self, global_info, max_num=1, decision_mode="multi"):
        messages = self._build_messages(global_info, max_num, decision_mode)
        logger.info(f"LLM scheduler prepared {len(messages)} messages with backend={self.backend}")
        if self.backend == "vllm":
            response_text = self._generate_with_vllm(messages)
        else:
            response_text = self._generate_with_transformers(messages)
        logger.info(f"LLM scheduler raw response: {response_text}")
        parsed = self._parse_json_block(response_text)
        selected_agents = self._extract_selected_agents(response_text, decision_mode, max_num=max_num, parsed=parsed)
        self._log_scheduler_exchange(
            global_info,
            messages,
            response_text,
            max_num,
            decision_mode,
            selected_agents=selected_agents,
            parsed=parsed,
            vllm_hp=self._get_vllm_hp() if self.backend == "vllm" else None,
        )

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
    def __init__(self, agent_graph, action_graph, config_path="config/policy.json"):
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
        self.policy_llm_config = self.config.get("policy_llm", {})
        scheduler_config = self.config.get("llm_scheduler", {})
        self.scheduler_config = scheduler_config
        self.scheduler_mode = scheduler_config.get("mode", "multi")
        self.scheduler_default_top_k = scheduler_config.get("top_k", min(self.max_num_agents, self.max_path))
        self.scheduler_confidence_temperature = max(
            float(scheduler_config.get("confidence_temperature", 0.15)), 1e-3
        )
        self.scheduler_top1_threshold = float(scheduler_config.get("top1_activation_threshold", 0.5))
        self.scheduler_margin_threshold = float(scheduler_config.get("top1_margin_threshold", 0.15))
        self.scheduler_top2_threshold = float(scheduler_config.get("top2_cumulative_threshold", 0.75))
        self.scheduler_top3_threshold = float(scheduler_config.get("top3_cumulative_threshold", 0.9))
        effective_scheduler_model_config = dict(scheduler_config)
        if "model_name" not in effective_scheduler_model_config and self.policy_llm_config.get("model_name"):
            effective_scheduler_model_config["model_name"] = self.policy_llm_config.get("model_name")
        self.llm_scheduler = LLM_Scheduler(self.agent_graph, self.action_graph, effective_scheduler_model_config)
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
        logger.info("step cost: {}".format(step_cost))
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
            logger.info(f"Model saved successfully to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None

    def load_model(self, path, strict=True):
        try:
            if not path or not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False

            checkpoint = torch.load(path, map_location="cpu")
            if "config" in checkpoint and isinstance(checkpoint["config"], dict):
                self.config.update({k: v for k, v in checkpoint["config"].items() if k not in self.config})

            logger.info(f"Model loaded successfully from {path}")
            logger.info(f"Model timestamp: {checkpoint.get('timestamp')}")
            if checkpoint.get("metadata", {}).get("tag"):
                logger.info(f"Model tag: {checkpoint['metadata']['tag']}")
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

    def _sharpen_scheduler_decisions(self, decisions):
        if not decisions:
            return []

        if len(decisions) == 1:
            only_decision = dict(decisions[0])
            only_decision["raw_confidence"] = float(only_decision.get("confidence", 1.0))
            only_decision["confidence"] = 1.0
            return [only_decision]

        raw_confidences = torch.tensor(
            [max(float(item.get("confidence", 0.0)), 1e-6) for item in decisions],
            device=self.device,
            dtype=torch.float32,
        )
        sharpened_confidences = torch.softmax(
            torch.log(raw_confidences + 1e-8) / self.scheduler_confidence_temperature,
            dim=0,
        )

        sharpened_decisions = []
        for item, raw_confidence, sharpened_confidence in zip(
            decisions,
            raw_confidences.tolist(),
            sharpened_confidences.tolist(),
        ):
            sharpened_item = dict(item)
            sharpened_item["raw_confidence"] = float(raw_confidence)
            sharpened_item["confidence"] = float(sharpened_confidence)
            sharpened_decisions.append(sharpened_item)

        sharpened_decisions = sorted(sharpened_decisions, key=lambda item: item["confidence"], reverse=True)
        return sharpened_decisions

    def _determine_scheduler_keep_k(self, decisions, max_num):
        if not decisions:
            return 0

        if self.scheduler_mode == "single_best":
            return 1

        candidate_count = min(len(decisions), max_num)
        candidate_confidences = [float(item["confidence"]) for item in decisions[:candidate_count]]

        if candidate_count == 1:
            return 1

        top1_confidence = candidate_confidences[0]
        top2_confidence = candidate_confidences[1]
        if (
            top1_confidence >= self.scheduler_top1_threshold
            or top1_confidence - top2_confidence >= self.scheduler_margin_threshold
        ):
            return 1

        if candidate_count >= 2 and sum(candidate_confidences[:2]) >= self.scheduler_top2_threshold:
            return 2

        if candidate_count >= 3 and sum(candidate_confidences[:3]) >= self.scheduler_top3_threshold:
            return 3

        return candidate_count

    def _decide_agent_indices(self, global_info):
        if self.scheduler_mode == "single_best":
            max_num = 1
        else:
            max_num = max(1, min(self.scheduler_default_top_k, self.max_num_agents, self.max_path))
        try:
            decisions = self.llm_scheduler.forward(global_info, max_num=max_num, decision_mode=self.scheduler_mode)
        except Exception as exc:
            if isinstance(exc, RetryError) and exc.last_attempt is not None:
                try:
                    root_exc = exc.last_attempt.exception()
                except Exception:
                    root_exc = None
                if root_exc is not None:
                    logger.warning(f"LLM policy failed: {exc}. Root cause: {root_exc}. Defaulting to terminator.")
                else:
                    logger.warning(f"LLM policy failed: {exc}. Defaulting to terminator.")
            else:
                logger.warning(f"LLM policy failed: {exc}. Defaulting to terminator.")
            fallback = [{"name": self.agent_role_list[self.end_action.item()], "index": self.end_action.item(), "confidence": 1.0}]
            return torch.tensor([self.end_action.item()], device=self.device), fallback

        decisions = self._sharpen_scheduler_decisions(decisions)
        keep_k = self._determine_scheduler_keep_k(decisions, max_num)
        decisions = decisions[:keep_k]
        logger.info(
            "Scheduler sharpened decisions: {}".format(
                [
                    {
                        "name": item["name"],
                        "raw_confidence": round(float(item.get("raw_confidence", item["confidence"])), 4),
                        "sharpened_confidence": round(float(item["confidence"]), 4),
                    }
                    for item in decisions
                ]
            )
        )
        logger.info("Scheduler keep_k after confidence sharpening: {}".format(keep_k))

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
        logger.info("Agent Indices: {}".format(agent_indices))
        return [self.agent_hash_list[i] for i in agent_indices]

    def init_forward(self, global_info):
        logger.info("[LLM Only] Init Policy Forward")
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
        logger.info("[LLM Only] Following Policy Forward")
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
        logger.info(f"trajectory[{trajectory_idx}]: {self.current_trajectories[trajectory_idx]}")

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
        logger.info("Inference mode: Skipping Update")
        self.current_trajectories = []
        self.executed_trajectories = []
        self.execution_count = 0
        return {}

    def finalize_task(self, transition, global_info):
        logger.info("transition reward: {}".format(transition.get("reward", 0)))
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
                logger.info("token cost: {}".format(cost))
                logger.info("cost factor: {}".format(cost / 100000))
                current_trajectory[index]["reward"] *= cost / 100000
                logger.info("Reward: {}".format(current_trajectory[index]["reward"]))

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
                    logger.info("Last Reward: {}".format(current_trajectory[-1]["reward"]))
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
                    logger.info("Last Reward: {}".format(current_trajectory[-1]["reward"]))
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
            logger.error(f"Error finding latest model: {str(e)}")
            return None
