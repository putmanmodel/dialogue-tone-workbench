from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from modules.generation import build_framing, detect_utterance_type


BLOCKS_DIR = Path("data/blocks")
CORPUS_EXCLUSIONS_PATH = BLOCKS_DIR / "excluded_entries.json"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "with",
    "you",
    "your",
}


def discover_block_files(blocks_dir: Path | None = None) -> List[Path]:
    root = blocks_dir or BLOCKS_DIR
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if path.name != CORPUS_EXCLUSIONS_PATH.name)


def load_corpus_exclusions(blocks_dir: Path | None = None) -> dict[str, set[int]]:
    root = blocks_dir or BLOCKS_DIR
    path = root / CORPUS_EXCLUSIONS_PATH.name
    if not path.exists():
        return {}

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    exclusions: dict[str, set[int]] = {}

    if isinstance(raw_data, dict) and isinstance(raw_data.get("entries"), list):
        reviewed_entries = raw_data["entries"]
        for item in reviewed_entries:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "exclude_dtw":
                continue
            file_name = item.get("source_block_file")
            index = item.get("index")
            if not isinstance(file_name, str) or not isinstance(index, int):
                continue
            exclusions.setdefault(file_name, set()).add(index)
        return exclusions

    if not isinstance(raw_data, dict):
        return exclusions

    for file_name, values in raw_data.items():
        if not isinstance(file_name, str) or not isinstance(values, list):
            continue
        exclusions[file_name] = {int(value) for value in values if isinstance(value, int)}
    return exclusions


def load_block_entries(path: Path) -> tuple[list[dict], str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"Could not load {path.name}: {exc}"

    exclusions = load_corpus_exclusions(path.parent)
    entries = extract_entries(data, excluded_indices=exclusions.get(path.name, set()))
    if not entries:
        return [], f"{path.name} does not contain any readable sentence entries."
    return entries, None


def extract_entries(data: object, excluded_indices: set[int] | None = None) -> list[dict]:
    candidates: list[object] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                candidates = value
                break

    excluded_indices = excluded_indices or set()
    entries: list[dict] = []
    for index, item in enumerate(candidates):
        if index in excluded_indices:
            continue
        if isinstance(item, dict):
            text = first_text_value(item, ["sentence", "text", "line", "base_line"])
            if text:
                entry = dict(item)
                entry["_source_entry_index"] = index
                entries.append(entry)
        elif isinstance(item, str) and item.strip():
            entries.append({"sentence": item.strip(), "_source_entry_index": index})
    return entries


def first_text_value(item: dict, keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def corpus_entry_text(entry: dict) -> str:
    return first_text_value(entry, ["sentence", "text", "line", "base_line"])


def load_corpus_bank(blocks_dir: Path | None = None) -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    errors: list[str] = []

    for path in discover_block_files(blocks_dir):
        file_entries, error = load_block_entries(path)
        if error:
            errors.append(error)
            continue
        for index, entry in enumerate(file_entries):
            entries.append(
                {
                    "text": corpus_entry_text(entry),
                    "source_block_file": path.name,
                    "source_entry_index": int(entry.get("_source_entry_index", index)),
                    "metadata": entry,
                }
            )

    return entries, errors


def normalize_for_generation(raw_input: str, blocks_dir: Path | None = None) -> Dict[str, object]:
    raw_clean = clean_text(raw_input)
    if not raw_clean:
        return {
            "raw_input": "",
            "normalized_generation_input": "",
            "normalization_applied": False,
            "normalization_mode": "none",
            "normalization_source_text": "",
            "normalization_source_block_file": "",
            "normalization_confidence": 0.0,
        }

    raw_utterance_type = detect_utterance_type(raw_clean, build_framing())

    if raw_utterance_type != "generic_statement":
        return {
            "raw_input": raw_clean,
            "normalized_generation_input": raw_clean,
            "normalization_applied": False,
            "normalization_mode": "none",
            "normalization_source_text": "",
            "normalization_source_block_file": "",
            "normalization_confidence": 1.0,
        }

    if is_interpersonally_ready(raw_clean):
        return {
            "raw_input": raw_clean,
            "normalized_generation_input": raw_clean,
            "normalization_applied": False,
            "normalization_mode": "none",
            "normalization_source_text": "",
            "normalization_source_block_file": "",
            "normalization_confidence": 1.0,
        }

    if starts_with_greeting(raw_clean) or mentions_testing(raw_clean) or asks_time(raw_clean):
        rewritten_text, confidence = rule_based_rewrite(raw_clean)
        return {
            "raw_input": raw_clean,
            "normalized_generation_input": rewritten_text,
            "normalization_applied": rewritten_text != raw_clean,
            "normalization_mode": "rule_based" if rewritten_text != raw_clean else "none",
            "normalization_source_text": "",
            "normalization_source_block_file": "",
            "normalization_confidence": confidence,
        }

    corpus_match = find_best_corpus_match(raw_clean, blocks_dir=blocks_dir)
    if corpus_match and should_accept_corpus_match(raw_clean, raw_utterance_type, corpus_match):
        return {
            "raw_input": raw_clean,
            "normalized_generation_input": corpus_match["text"],
            "normalization_applied": True,
            "normalization_mode": "corpus_match",
            "normalization_source_text": corpus_match["text"],
            "normalization_source_block_file": corpus_match["source_block_file"],
            "normalization_confidence": round(corpus_match["score"], 2),
        }

    rewritten_text, confidence = rule_based_rewrite(raw_clean)
    return {
        "raw_input": raw_clean,
        "normalized_generation_input": rewritten_text,
        "normalization_applied": rewritten_text != raw_clean,
        "normalization_mode": "rule_based" if rewritten_text != raw_clean else "none",
        "normalization_source_text": "",
        "normalization_source_block_file": "",
        "normalization_confidence": confidence,
    }


def is_interpersonally_ready(text: str) -> bool:
    lowered = text.lower().strip()
    flat_exact = {
        "hello",
        "hi",
        "hey",
        "testing",
        "testing this",
        "test",
        "does this work?",
        "does this work",
        "what time is it?",
        "what time is it",
        "ok",
        "okay",
    }
    if lowered in flat_exact:
        return False

    tokens = tokenize(lowered)
    if len(tokens) <= 2:
        return False

    interpersonal_markers = {
        "i",
        "you",
        "we",
        "me",
        "us",
        "sorry",
        "meant",
        "trying",
        "need",
        "want",
        "feel",
        "heard",
        "understand",
        "talk",
    }
    if any(token in interpersonal_markers for token in tokens) and len(tokens) >= 4:
        return True

    if "'" in text and len(tokens) >= 4:
        return True

    if text.endswith("?") and any(token in {"i", "you", "we"} for token in tokens):
        return True

    return False


def find_best_corpus_match(raw_input: str, blocks_dir: Path | None = None) -> Dict[str, object] | None:
    corpus_bank, _errors = load_corpus_bank(blocks_dir)
    if not corpus_bank:
        return None

    raw_utterance_type = detect_utterance_type(raw_input, build_framing())
    raw_tokens = meaningful_tokens(raw_input)
    raw_bigrams = bigrams(tokenize(raw_input))
    raw_lower = raw_input.lower()

    best_match = None
    best_score = 0.0

    for candidate in corpus_bank:
        candidate_text = candidate["text"]
        candidate_lower = candidate_text.lower()
        candidate_tokens = meaningful_tokens(candidate_text)
        candidate_bigrams = bigrams(tokenize(candidate_text))

        token_overlap = overlap_ratio(raw_tokens, candidate_tokens)
        token_jaccard = jaccard_ratio(raw_tokens, candidate_tokens)
        phrase_overlap = jaccard_ratio(raw_bigrams, candidate_bigrams)
        candidate_utterance_type = detect_utterance_type(candidate_text, build_framing())

        score = (0.5 * token_overlap) + (0.3 * token_jaccard) + (0.2 * phrase_overlap)

        if raw_lower in candidate_lower or candidate_lower in raw_lower:
            score += 0.18
        if raw_input.endswith("?") and candidate_text.endswith("?"):
            score += 0.06
        if starts_with_greeting(raw_input) and starts_with_greeting(candidate_text):
            score += 0.08
        if mentions_testing(raw_input) and mentions_testing(candidate_text):
            score += 0.12
        if raw_utterance_type == candidate_utterance_type:
            score += 0.1
        if raw_input.endswith("?") != candidate_text.endswith("?"):
            score -= 0.18
        if not same_semantic_family(raw_utterance_type, candidate_utterance_type):
            score -= 0.25

        if score > best_score:
            best_score = score
            best_match = {
                "text": candidate_text,
                "source_block_file": candidate["source_block_file"],
                "source_entry_index": candidate["source_entry_index"],
                "score": score,
                "token_overlap": token_overlap,
                "phrase_overlap": phrase_overlap,
                "candidate_utterance_type": candidate_utterance_type,
            }

    if not best_match:
        return None

    if best_match["token_overlap"] == 0 and best_match["phrase_overlap"] == 0:
        return None

    return best_match


def should_accept_corpus_match(raw_input: str, raw_utterance_type: str, match: Dict[str, object]) -> bool:
    candidate_utterance_type = str(match.get("candidate_utterance_type", "generic_statement"))
    score = float(match.get("score", 0.0))
    token_overlap = float(match.get("token_overlap", 0.0))
    phrase_overlap = float(match.get("phrase_overlap", 0.0))

    if not same_semantic_family(raw_utterance_type, candidate_utterance_type):
        return False
    if raw_input.endswith("?") != str(match.get("text", "")).endswith("?"):
        return False
    if token_overlap < 0.5 and phrase_overlap < 0.2:
        return False
    if score < 0.68:
        return False
    return True


def rule_based_rewrite(raw_input: str) -> tuple[str, float]:
    lowered = raw_input.lower().strip()

    if starts_with_greeting(raw_input):
        return "I'm just trying to start the conversation.", 0.62

    if mentions_testing(raw_input):
        return "I'm just checking whether this is coming through.", 0.76

    if asks_time(raw_input):
        return "I'm asking a simple question, not trying to start anything.", 0.71

    if raw_input.endswith("?"):
        return "I'm just asking a straightforward question.", 0.58

    if lowered in {"ok", "okay", "fine"}:
        return "I'm trying to keep this simple.", 0.52

    if lowered in {"thanks", "thank you"}:
        return "I'm just trying to acknowledge that.", 0.54

    if len(tokenize(raw_input)) <= 3:
        return "I'm trying to say this plainly.", 0.48

    return clean_text(raw_input), 0.35


def clean_text(text: str) -> str:
    collapsed = " ".join((text or "").strip().split())
    if not collapsed:
        return ""
    if collapsed[-1] not in ".!?":
        collapsed = f"{collapsed}."
    return collapsed


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def bigrams(tokens: list[str]) -> set[str]:
    return {" ".join(tokens[index : index + 2]) for index in range(len(tokens) - 1)}


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def jaccard_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def starts_with_greeting(text: str) -> bool:
    lowered = text.lower().strip()
    return lowered in {"hi", "hi.", "hello", "hello.", "hey", "hey."}


def mentions_testing(text: str) -> bool:
    lowered = text.lower()
    exact_or_prefix = [
        "test",
        "testing",
        "testing this",
        "does this work",
        "can you hear me",
        "is this working",
        "is this coming through",
        "just testing",
    ]
    return any(lowered == phrase or lowered.startswith(f"{phrase}?") or lowered.startswith(f"{phrase}.") for phrase in exact_or_prefix)


def asks_time(text: str) -> bool:
    lowered = text.lower()
    return "what time is it" in lowered or "what time" in lowered


def same_semantic_family(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == "generic_statement" and right == "generic_statement":
        return True
    if left == "request_or_question" and right == "request_or_question":
        return True
    if left == "clarification" and right == "clarification":
        return True
    return False
