import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tasks.evaluator import BenchmarkEvaluator


def _normalize_levels(level):
    if isinstance(level, str):
        token = level.strip().lower()
        if token == "all":
            return [1, 2, 3]
        if token.isdigit():
            return [int(token)]
        raise ValueError(f"Unsupported GAIA level value: {level}")
    if isinstance(level, int):
        if level <= 0:
            raise ValueError(f"Unsupported GAIA level value: {level}")
        return [level]
    raise ValueError(f"Unsupported GAIA level value: {level}")


def _candidate_metadata_paths(data_dir, mode, level):
    roots = [
        Path(data_dir) / "2023" / mode,
        Path(data_dir) / mode,
    ]

    filenames = []
    if level is not None and level > 0:
        filenames.extend(
            [
                f"metadata.level{level}.parquet",
                f"metadata.level{level}.jsonl",
            ]
        )
    filenames.extend(["metadata.parquet", "metadata.jsonl"])

    for root in roots:
        for filename in filenames:
            yield root / filename


def _load_metadata(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported GAIA metadata format: {path}")


def load_dataset(mode, level=1, data_limit=None, data_dir="data/GAIA"):
    metadata_path = None
    for candidate in _candidate_metadata_paths(data_dir, mode, level):
        if candidate.exists():
            metadata_path = candidate
            break

    if metadata_path is None:
        searched = "\n".join(str(path) for path in _candidate_metadata_paths(data_dir, mode, level))
        raise FileNotFoundError(f"Unable to locate GAIA metadata file. Searched:\n{searched}")

    data = _load_metadata(metadata_path)
    data.attrs["gaia_base_dir"] = str(metadata_path.parent)
    return data[:data_limit] if data_limit else data


def _pick_value(row, *keys, default=None):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def format_question(row, idx, base_dir):
    question = _pick_value(row, "Question", "question", "prompt", default="")
    answer = _pick_value(row, "Final answer", "final_answer", "Answer", "answer")
    task_id = _pick_value(row, "task_id", "Task ID", "id", default=idx)
    level = _pick_value(row, "Level", "level")
    file_name = _pick_value(row, "file_name", "file", "attachment", "attachments")

    file_relative_path = None
    if isinstance(file_name, str) and file_name.strip():
        attachment_path = (Path(base_dir) / file_name).resolve()
        data_root = (Path("data")).resolve()
        try:
            file_relative_path = attachment_path.relative_to(data_root).as_posix()
        except ValueError:
            file_relative_path = attachment_path.as_posix()

    task = {
        "type": "GAIA",
        "Question": question,
        "Answer": answer,
        "id": task_id,
        "level": int(level) if level is not None and str(level).isdigit() else level,
        "file_name": file_relative_path,
        "metadata": row.to_dict(),
    }
    return task


def run(runner, evaluator, results_dir, mode, data_limit=None, level=1, data_dir="data/GAIA"):
    levels = _normalize_levels(level)
    for current_level in levels:
        dataset = load_dataset(mode=mode, level=current_level, data_limit=data_limit, data_dir=data_dir)
        base_dir = Path(dataset.attrs.get("gaia_base_dir", Path(data_dir) / "2023" / mode))
        result_path = Path(results_dir) / f"GAIA_{mode}_level{current_level}.jsonl"

        with open(result_path, "w", encoding="utf-8") as fd:
            for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
                task = format_question(row, idx, base_dir)
                final_ans = runner.run_reasoning(task)
                flag = None
                if task.get("Answer") is not None:
                    flag = BenchmarkEvaluator.check_gaia(final_ans, task["Answer"])
                record = {
                    "id": task["id"],
                    "level": task.get("level"),
                    "pred": final_ans,
                    "correct": flag,
                    "file_name": task.get("file_name"),
                }
                fd.write(json.dumps(record, ensure_ascii=False) + "\n")
