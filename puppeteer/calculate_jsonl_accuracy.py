import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate accuracy from JSONL result files under a directory.")
    parser.add_argument(
        "data_root",
        type=str,
        help="Directory containing result JSONL files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for JSONL files under the input directory.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file trajectory accuracy.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=0,
        help="Show up to N failed final trajectories.",
    )
    parser.add_argument(
        "--show-changed",
        type=int,
        default=0,
        help="Show up to N trajectories whose recent_answers changed along the path.",
    )
    return parser.parse_args()


def iter_jsonl_files(data_root: str, recursive: bool) -> Iterable[str]:
    if recursive:
        for root, _, files in os.walk(data_root):
            for file_name in sorted(files):
                if file_name.endswith(".jsonl"):
                    yield os.path.join(root, file_name)
        return

    for file_name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, file_name)
        if os.path.isfile(path) and file_name.endswith(".jsonl"):
            yield path


def load_final_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

            outcome = record.get("outcome") or {}
            if outcome.get("done") is not True:
                continue

            task = record.get("task") or {}
            state = record.get("state") or {}
            metadata = record.get("metadata") or {}
            recent_answers = state.get("recent_answers") or []
            pred = recent_answers[-1] if recent_answers else metadata.get("answer_summary")

            records.append(
                {
                    "file": path,
                    "file_name": os.path.basename(path),
                    "episode_id": record.get("episode_id"),
                    "path_id": record.get("path_id"),
                    "task_id": task.get("id"),
                    "t": record.get("t", -1),
                    "ground_truth": task.get("Answer"),
                    "prediction": pred,
                    "recent_answers": list(recent_answers),
                    "correct": pred == task.get("Answer"),
                }
            )
    return records


def deduplicate_trajectories(records: List[Dict]) -> List[Dict]:
    latest_by_trajectory: Dict[tuple, Dict] = {}
    for record in sorted(records, key=lambda item: (str(item["episode_id"]), str(item["path_id"]), int(item["t"]))):
        key = (record["episode_id"], record["path_id"])
        latest_by_trajectory[key] = record
    return list(latest_by_trajectory.values())


def round_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def summarize(records: List[Dict]) -> Dict[str, object]:
    trajectory_records = deduplicate_trajectories(records)
    by_task: Dict[object, List[Dict]] = defaultdict(list)
    for record in trajectory_records:
        by_task[record["task_id"]].append(record)

    task_last = {}
    for task_id, task_records in by_task.items():
        task_last[task_id] = sorted(
            task_records,
            key=lambda item: (item["file_name"], str(item["episode_id"]), str(item["path_id"]), int(item["t"])),
        )[-1]

    trajectory_correct = sum(1 for record in trajectory_records if record["correct"])
    unchanged_trajectories = sum(
        1
        for record in trajectory_records
        if record["recent_answers"] and len(set(record["recent_answers"])) == 1
    )
    task_last_correct = sum(1 for record in task_last.values() if record["correct"])
    task_any_correct = sum(1 for records_for_task in by_task.values() if any(record["correct"] for record in records_for_task))

    return {
        "final_done_records": len(records),
        "trajectory_records": trajectory_records,
        "completed_trajectories": len(trajectory_records),
        "correct_trajectories": trajectory_correct,
        "trajectory_accuracy": round_pct(trajectory_correct, len(trajectory_records)),
        "unchanged_trajectories": unchanged_trajectories,
        "unchanged_ratio": round_pct(unchanged_trajectories, len(trajectory_records)),
        "unique_tasks": len(by_task),
        "correct_tasks_last": task_last_correct,
        "task_accuracy_last": round_pct(task_last_correct, len(task_last)),
        "correct_tasks_any": task_any_correct,
        "task_accuracy_any": round_pct(task_any_correct, len(by_task)),
    }


def summarize_per_file(records: List[Dict]) -> List[Dict]:
    per_file: Dict[str, List[Dict]] = defaultdict(list)
    for record in deduplicate_trajectories(records):
        per_file[record["file_name"]].append(record)

    rows: List[Dict] = []
    for file_name in sorted(per_file):
        file_records = per_file[file_name]
        correct = sum(1 for record in file_records if record["correct"])
        rows.append(
            {
                "file_name": file_name,
                "completed_trajectories": len(file_records),
                "correct_trajectories": correct,
                "trajectory_accuracy": round_pct(correct, len(file_records)),
            }
        )
    return rows


def print_summary(summary: Dict[str, object], file_count: int) -> None:
    print(f"Scanned files: {file_count}")
    print(f"Final done records: {summary['final_done_records']}")
    print(
        f"Trajectory accuracy: {summary['correct_trajectories']}/{summary['completed_trajectories']} "
        f"= {summary['trajectory_accuracy']}%"
    )
    print(
        f"Task accuracy (last result): {summary['correct_tasks_last']}/{summary['unique_tasks']} "
        f"= {summary['task_accuracy_last']}%"
    )
    print(
        f"Task accuracy (any correct): {summary['correct_tasks_any']}/{summary['unique_tasks']} "
        f"= {summary['task_accuracy_any']}%"
    )
    print(
        f"Unchanged-answer trajectories: {summary['unchanged_trajectories']}/{summary['completed_trajectories']} "
        f"= {summary['unchanged_ratio']}%"
    )


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Input directory not found: {args.data_root}")

    file_paths = list(iter_jsonl_files(args.data_root, recursive=args.recursive))
    if not file_paths:
        raise FileNotFoundError(f"No JSONL files found under: {args.data_root}")

    all_records: List[Dict] = []
    for path in file_paths:
        all_records.extend(load_final_records(path))

    summary = summarize(all_records)
    print_summary(summary, file_count=len(file_paths))

    if args.show_files:
        print("\nPer-file trajectory accuracy:")
        for row in summarize_per_file(all_records):
            print(
                f"- {row['file_name']} | completed={row['completed_trajectories']} "
                f"| correct={row['correct_trajectories']} | accuracy={row['trajectory_accuracy']}%"
            )

    if args.show_failures > 0:
        failures = [record for record in summary["trajectory_records"] if not record["correct"]]
        print(f"\nFailed trajectories (showing up to {args.show_failures}):")
        for record in failures[: args.show_failures]:
            print(
                f"- file={record['file_name']} | task_id={record['task_id']} | episode_id={record['episode_id']} "
                f"| path_id={record['path_id']} | gt={record['ground_truth']} | pred={record['prediction']}"
            )

    if args.show_changed > 0:
        changed = [
            record
            for record in summary["trajectory_records"]
            if not record["recent_answers"] or len(set(record["recent_answers"])) > 1
        ]
        print(f"\nChanged-answer trajectories (showing up to {args.show_changed}):")
        for record in changed[: args.show_changed]:
            print(
                f"- file={record['file_name']} | task_id={record['task_id']} | episode_id={record['episode_id']} "
                f"| path_id={record['path_id']} | recent_answers={record['recent_answers']}"
            )


if __name__ == "__main__":
    main()
