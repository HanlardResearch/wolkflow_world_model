import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path(
    r"D:\Research_HUB\ChatDev\puppeteer\results\world_model_dataset-llm\GAIA\test\workflow_world_model_20260409_151942.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a JSONL file and convert all records into structured text."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path for the structured text output file.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number}: {exc.msg}"
                ) from exc
    return records


def format_scalar(value: Any) -> str:
    if isinstance(value, str):
        if "\n" in value:
            return value
        return value
    return json.dumps(value, ensure_ascii=False)


def format_value(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent

    if isinstance(value, dict):
        if not value:
            return [f"{pad}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(format_value(item, indent + 2))
            else:
                item_text = format_scalar(item)
                if "\n" in item_text:
                    lines.append(f"{pad}{key}: |")
                    for line in item_text.splitlines():
                        lines.append(f"{' ' * (indent + 2)}{line}")
                else:
                    lines.append(f"{pad}{key}: {item_text}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{pad}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(format_value(item, indent + 2))
            else:
                item_text = format_scalar(item)
                if "\n" in item_text:
                    lines.append(f"{pad}- |")
                    for line in item_text.splitlines():
                        lines.append(f"{' ' * (indent + 2)}{line}")
                else:
                    lines.append(f"{pad}- {item_text}")
        return lines

    return [f"{pad}{format_scalar(value)}"]


def build_structured_text(records: list[dict[str, Any]], source_path: Path) -> str:
    lines = [
        "JSONL Structured Text",
        f"Source: {source_path}",
        f"Total records: {len(records)}",
        "",
    ]

    for index, record in enumerate(records, start=1):
        lines.append(f"===== Record {index} =====")
        lines.extend(format_value(record))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_suffix(".structured.txt")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = resolve_output_path(input_path, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    structured_text = build_structured_text(records, input_path)
    output_path.write_text(structured_text, encoding="utf-8")

    print(f"Read {len(records)} records from: {input_path}")
    print(f"Structured text written to: {output_path}")


if __name__ == "__main__":
    main()
