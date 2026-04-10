#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference.policy.workflow_world_model import QwenTextEmbeddingEncoder
from train_workflow_world_model import find_dataset_files
from utils.file_utils import iter_jsonl, write_jsonl


TextItem = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Qwen text embeddings into a JSONL cache.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset-filename", type=str, default="workflow_world_model")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def _stable_shard(prompt_name: str, text: str, num_shards: int) -> int:
    payload = f"{prompt_name}\n{text}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest, 16) % max(num_shards, 1)


def _step_text_fields(step: Dict) -> List[Tuple[str, str]]:
    return [
        ("", str(step.get("parameter", ""))),
        ("", str(step.get("answer_summary", ""))),
        ("", str(step.get("step_data_summary", ""))),
        ("", str(step.get("raw_prompt", ""))),
        ("", str(step.get("raw_response", ""))),
        ("", str(step.get("system_prompt", ""))),
    ]


def _iter_record_texts(record: Dict) -> Iterable[TextItem]:
    task = record.get("task", {})
    question = str(task.get("question", task.get("prompt", task.get("instruction", ""))))
    if question.strip():
        yield ("", question)

    state = record.get("state", {})
    steps = state.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for prompt_name, text in _step_text_fields(step):
                if text.strip():
                    yield (prompt_name, text)

    evidence_groups = [
        state.get("reasoning_results", []),
        state.get("tool_results", []),
        state.get("recent_answers", []),
    ]
    for group in evidence_groups:
        if not isinstance(group, list):
            continue
        for item in group:
            text = str(item or "")
            if text.strip():
                yield ("", text)


def collect_texts(records: Sequence[Dict], shard_index: int, num_shards: int) -> List[TextItem]:
    seen: Set[TextItem] = set()
    for record in records:
        for prompt_name, text in _iter_record_texts(record):
            item = (prompt_name, text)
            if item in seen:
                continue
            if _stable_shard(prompt_name, text, num_shards) != shard_index:
                continue
            seen.add(item)
    return sorted(seen)


def load_records(paths: Sequence[str], max_records: int) -> List[Dict]:
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


def main() -> None:
    args = parse_args()
    paths = find_dataset_files(args.data_root, args.dataset_filename, args.max_files)
    if not paths:
        raise FileNotFoundError(
            f"No dataset JSONL files matching '{args.dataset_filename}' were found under '{args.data_root}'."
        )
    records = load_records(paths, args.max_records)
    texts = collect_texts(records, args.shard_index, args.num_shards)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    encoder = QwenTextEmbeddingEncoder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        devices=[args.device],
    )
    try:
        with open(args.output_path, "w", encoding="utf-8") as handle:
            batch_size = max(int(args.batch_size), 1)
            total = len(texts)
            encoded_count = 0
            prompt_names = sorted({prompt_name for prompt_name, _ in texts})
            for current_prompt_name in prompt_names:
                prompt_texts = [item for item in texts if item[0] == current_prompt_name]
                for start in range(0, len(prompt_texts), batch_size):
                    batch = prompt_texts[start : start + batch_size]
                    embeddings = encoder.encode(
                        [text for _, text in batch],
                        prompt_name=current_prompt_name or None,
                    )
                    for (item_prompt_name, text), embedding in zip(batch, embeddings):
                        write_jsonl(
                            handle,
                            {
                                "prompt_name": item_prompt_name,
                                "text": text,
                                "embedding": embedding,
                            },
                        )
                    encoded_count += len(batch)
                    print(
                        f"[Shard {args.shard_index}/{args.num_shards}] encoded {encoded_count}/{total} texts on {args.device}",
                        flush=True,
                    )
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
