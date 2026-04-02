import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count world-model dataset samples from recorder JSONL files.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="results/world_model_dataset",
        help="Root directory to search for world-model dataset JSONL files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="workflow_world_model",
        help="Substring that must appear in dataset filenames.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Optional limit on the number of files to scan.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file sample counts.",
    )
    return parser.parse_args()


def iter_dataset_files(data_root: str, filename_substring: str) -> Iterable[str]:
    for root, _, files in os.walk(data_root):
        for file_name in sorted(files):
            if filename_substring in file_name and file_name.endswith(".jsonl"):
                yield os.path.join(root, file_name)


def count_file(path: str) -> Dict:
    sample_count = 0
    episode_ids = set()
    path_ids = Counter()
    action_names = Counter()
    task_types = Counter()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sample_count += 1
            try:
                record = json.loads(line)
            except Exception:
                continue

            episode_id = record.get("episode_id")
            if episode_id is not None:
                episode_ids.add(str(episode_id))

            path_id = record.get("path_id")
            if path_id is not None:
                path_ids[str(path_id)] += 1

            action_name = record.get("action", {}).get("name")
            if action_name:
                action_names[str(action_name)] += 1

            task = record.get("task", {})
            task_type = task.get("task_type") or task.get("type") or task.get("dataset_name")
            if task_type:
                task_types[str(task_type)] += 1

    return {
        "path": path,
        "samples": sample_count,
        "episodes": len(episode_ids),
        "path_ids": path_ids,
        "action_names": action_names,
        "task_types": task_types,
    }


def merge_counters(items: List[Dict]) -> Dict:
    total_samples = 0
    total_episodes = 0
    merged_paths = Counter()
    merged_actions = Counter()
    merged_task_types = Counter()

    for item in items:
        total_samples += item["samples"]
        total_episodes += item["episodes"]
        merged_paths.update(item["path_ids"])
        merged_actions.update(item["action_names"])
        merged_task_types.update(item["task_types"])

    return {
        "samples": total_samples,
        "episodes": total_episodes,
        "path_ids": merged_paths,
        "action_names": merged_actions,
        "task_types": merged_task_types,
    }


def format_top(counter: Counter, limit: int = 10) -> str:
    if not counter:
        return "None"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def main() -> None:
    args = parse_args()
    file_paths = list(iter_dataset_files(args.data_root, args.filename))
    if args.max_files > 0:
        file_paths = file_paths[: args.max_files]

    if not file_paths:
        raise FileNotFoundError(
            f"No JSONL files containing '{args.filename}' were found under '{args.data_root}'."
        )

    file_stats = [count_file(path) for path in file_paths]
    merged = merge_counters(file_stats)

    print(f"Scanned files: {len(file_stats)}")
    print(f"Total samples: {merged['samples']}")
    print(f"Total episodes: {merged['episodes']}")
    print(f"Path distribution: {format_top(merged['path_ids'])}")
    print(f"Top actions: {format_top(merged['action_names'])}")
    print(f"Top task types: {format_top(merged['task_types'])}")

    if args.show_files:
        print("\nPer-file counts:")
        for item in file_stats:
            print(f"- {item['path']} | samples={item['samples']} | episodes={item['episodes']}")


if __name__ == "__main__":
    main()
