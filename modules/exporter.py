from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from modules.generation import normalize_line
from modules.tagging import build_context_flags


OUTPUTS_DIR = Path("outputs")
DATASET_PATH = OUTPUTS_DIR / "dataset.jsonl"
APP_VERSION = "0.1.4"
GENERATION_MODE = "utterance_type_paraphrase_rules_v1"


def ensure_output_paths() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.touch(exist_ok=True)


def build_payload(
    base_line: str,
    speaker: str,
    target: str,
    context: str,
    variants: list[Dict[str, object]],
    input_mode: str = "manual",
    source_block_file: str | None = None,
    source_entry_index: int | None = None,
    raw_input: str | None = None,
    normalized_generation_input: str | None = None,
    normalization_applied: bool = False,
    normalization_mode: str = "none",
    normalization_source_text: str | None = None,
    normalization_source_block_file: str | None = None,
    normalization_confidence: float = 0.0,
    utterance_type: str = "generic_statement",
    generation_strategy: str | None = None,
) -> Dict[str, object]:
    normalized_base = normalize_line(base_line)
    return {
        "version": APP_VERSION,
        "generation_mode": GENERATION_MODE,
        "generation_strategy": generation_strategy or GENERATION_MODE,
        "timestamp": current_timestamp(),
        "base_line": base_line.strip(),
        "base_line_normalized": normalized_base,
        "speaker": speaker.strip(),
        "target": target.strip(),
        "context": context.strip(),
        "input_mode": input_mode,
        "source_block_file": source_block_file or "",
        "source_entry_index": source_entry_index,
        "raw_input": (raw_input if raw_input is not None else base_line).strip(),
        "normalized_generation_input": (normalized_generation_input if normalized_generation_input is not None else normalized_base).strip(),
        "normalization_applied": normalization_applied,
        "normalization_mode": normalization_mode,
        "normalization_source_text": normalization_source_text or "",
        "normalization_source_block_file": normalization_source_block_file or "",
        "normalization_confidence": round(float(normalization_confidence), 2),
        "utterance_type": utterance_type,
        "context_flags": build_context_flags(context),
        "variants": variants,
    }


def append_to_dataset(payload: Dict[str, object]) -> Path:
    ensure_output_paths()
    with DATASET_PATH.open("a", encoding="utf-8") as dataset_file:
        dataset_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return DATASET_PATH


def export_payload(payload: Dict[str, object]) -> Path:
    ensure_output_paths()
    timestamp_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    line_slug = slugify(str(payload.get("base_line_normalized", "")))
    filename = f"{timestamp_slug}--{line_slug or 'dialogue-run'}.json"
    export_path = OUTPUTS_DIR / filename

    with export_path.open("w", encoding="utf-8") as export_file:
        json.dump(payload, export_file, indent=2, ensure_ascii=False, sort_keys=False)

    return export_path


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    compact = "-".join(part for part in safe.split("-") if part)
    return compact[:50]
