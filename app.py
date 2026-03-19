from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import streamlit as st

from modules.exporter import append_to_dataset, build_payload, ensure_output_paths, export_payload
from modules.generation import VARIANT_CATEGORIES, generate_variants
from modules.normalization import corpus_entry_text, discover_block_files, load_block_entries, normalize_for_generation
from modules.tagging import tag_variant


st.set_page_config(
    page_title="Dialogue Tone Workbench",
    layout="wide",
)

ensure_output_paths()
BLOCKS_DIR = Path("data/blocks")
SESSION_DEFAULTS = {
    "current_payload": None,
    "input_mode_choice": "Manual Input",
    "manual_raw_input": "",
    "speaker_input": "",
    "target_input": "",
    "context_input": "",
    "corpus_selected_file": "",
    "corpus_entry_index": 0,
}


def ensure_session_defaults() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app_state() -> None:
    for key in SESSION_DEFAULTS:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def handle_corpus_block_change() -> None:
    st.session_state.corpus_entry_index = 0


def corpus_entry_options(entries: list[dict]) -> list[str]:
    options = []
    for index, entry in enumerate(entries):
        preview = corpus_entry_text(entry)
        preview = preview if len(preview) <= 80 else f"{preview[:77]}..."
        options.append(f"{index}: {preview}")
    return options


def generate_payload(
    raw_input: str,
    speaker: str,
    target: str,
    context: str,
    input_mode: str = "manual",
    source_block_file: str | None = None,
    source_entry_index: int | None = None,
) -> dict:
    normalization = normalize_for_generation(raw_input, blocks_dir=BLOCKS_DIR)
    generation_input = normalization["normalized_generation_input"]

    generation_result = generate_variants(
        base_line=generation_input,
        speaker=speaker,
        target=target,
        context=context,
    )

    tagged_variants = []
    for item in generation_result["variants"]:
        tags = tag_variant(text=item["text"], target_tone=item["target_tone"])
        tagged_variants.append(
            {
                "text": item["text"],
                "target_tone": tags["target_tone"],
                "detected_tone": tags["detected_tone"],
                "intensity": tags["intensity"],
                "polarity": tags["polarity"],
                "notes": tags["notes"],
            }
        )

    return build_payload(
        base_line=generation_input,
        speaker=speaker,
        target=target,
        context=context,
        variants=tagged_variants,
        input_mode=input_mode,
        source_block_file=source_block_file,
        source_entry_index=source_entry_index,
        raw_input=normalization["raw_input"],
        normalized_generation_input=generation_input,
        normalization_applied=bool(normalization["normalization_applied"]),
        normalization_mode=str(normalization["normalization_mode"]),
        normalization_source_text=str(normalization["normalization_source_text"]),
        normalization_source_block_file=str(normalization["normalization_source_block_file"]),
        normalization_confidence=float(normalization["normalization_confidence"]),
        utterance_type=str(generation_result["utterance_type"]),
        generation_strategy=str(generation_result["generation_strategy"]),
    )


def render_variant_card(index: int, variant: dict) -> None:
    st.markdown(f"### {index}. {variant['target_tone'].title()}")
    st.write(variant["text"])
    st.caption(
        f"Detected: {variant['detected_tone']} | "
        f"Intensity: {variant['intensity']:.2f} | "
        f"Polarity: {variant['polarity']}"
    )


def build_variant_rows(payload: dict) -> List[dict]:
    rows = []
    for variant in payload["variants"]:
        rows.append(
            {
                "target_tone": variant["target_tone"],
                "detected_tone": variant["detected_tone"],
                "intensity": variant["intensity"],
                "polarity": variant["polarity"],
                "text": variant["text"],
            }
        )
    return rows


def main() -> None:
    st.title("Dialogue Tone Workbench")
    st.write(
        "A deterministic, local-first workbench for exploring six controlled tone variations of a single line of dialogue."
    )
    st.caption("Deterministic generation. Fixed six-tone set. Local JSON and JSONL export.")

    ensure_session_defaults()

    left_col, center_col, right_col = st.columns([1.1, 1.5, 1.2], gap="large")

    with left_col:
        st.subheader("Source Line")
        input_mode_label = st.radio(
            "Input mode",
            options=["Manual Input", "Corpus Test Mode"],
            horizontal=False,
            key="input_mode_choice",
        )
        input_mode = "manual" if input_mode_label == "Manual Input" else "corpus"

        selected_block_file = ""
        selected_entry_index = None
        selected_entry_text = ""

        if input_mode == "manual":
            base_line = st.text_input("Input line", placeholder="I didn't mean it like that.", key="manual_raw_input")
        else:
            block_files = discover_block_files()
            if not BLOCKS_DIR.exists():
                st.info("Corpus Test Mode is available when block files exist under data/blocks/.")
                base_line = st.text_input("Input line", placeholder="I didn't mean it like that.", key="manual_raw_input")
            elif not block_files:
                st.info("No block JSON files were found in data/blocks/. Manual Input is still available.")
                base_line = st.text_input("Input line", placeholder="I didn't mean it like that.", key="manual_raw_input")
            else:
                file_names = [path.name for path in block_files]
                if st.session_state.corpus_selected_file not in file_names:
                    st.session_state.corpus_selected_file = file_names[0]

                selected_block_file = st.selectbox(
                    "Block file",
                    options=file_names,
                    key="corpus_selected_file",
                    on_change=handle_corpus_block_change,
                )

                selected_path = BLOCKS_DIR / selected_block_file
                entries, load_error = load_block_entries(selected_path)
                if load_error:
                    st.error(load_error)
                    base_line = st.text_input("Input line", placeholder="I didn't mean it like that.", key="manual_raw_input")
                elif not entries:
                    st.warning("This block file does not contain usable entries.")
                    base_line = st.text_input("Input line", placeholder="I didn't mean it like that.", key="manual_raw_input")
                else:
                    max_index = len(entries) - 1
                    st.session_state.corpus_entry_index = max(0, min(st.session_state.corpus_entry_index, max_index))

                    nav_prev, nav_next, nav_random = st.columns(3)
                    if nav_prev.button("Previous Entry", use_container_width=True):
                        st.session_state.corpus_entry_index = (st.session_state.corpus_entry_index - 1) % len(entries)
                    if nav_next.button("Next Entry", use_container_width=True):
                        st.session_state.corpus_entry_index = (st.session_state.corpus_entry_index + 1) % len(entries)
                    if nav_random.button("Random Entry", use_container_width=True):
                        st.session_state.corpus_entry_index = random.randrange(len(entries))

                    option_labels = corpus_entry_options(entries)
                    current_label = option_labels[st.session_state.corpus_entry_index]
                    selected_label = st.selectbox(
                        "Entry",
                        options=option_labels,
                        index=st.session_state.corpus_entry_index,
                    )
                    if selected_label != current_label:
                        st.session_state.corpus_entry_index = option_labels.index(selected_label)

                    filtered_entry_index = st.session_state.corpus_entry_index
                    selected_entry = entries[filtered_entry_index]
                    selected_entry_index = int(selected_entry.get("_source_entry_index", filtered_entry_index))
                    selected_entry_text = corpus_entry_text(selected_entry)
                    base_line = selected_entry_text

                    st.caption(f"Source file: {selected_block_file}")
                    st.caption(f"Selected entry index: {selected_entry_index}")
                    st.caption(f"Selected source text: {selected_entry_text}")
                    extra_metadata = {
                        key: value
                        for key, value in selected_entry.items()
                        if key not in {"sentence", "text", "line", "base_line"}
                    }
                    if extra_metadata:
                        st.json(extra_metadata)
                    st.text_area("Corpus line", value=selected_entry_text, disabled=True, height=100)

        speaker = st.text_input("Speaker (optional)", placeholder="Speaker", key="speaker_input")
        target = st.text_input("Target (optional)", placeholder="Target", key="target_input")
        context = st.text_area(
            "Context (optional)",
            placeholder="Short scene context or emotional setup.",
            height=120,
            key="context_input",
        )
        st.text_input(
            "Tone set",
            value=", ".join(VARIANT_CATEGORIES),
            disabled=True,
        )

        action_generate, action_export, action_reset = st.columns(3)
        generate_clicked = action_generate.button("Generate", type="primary", use_container_width=True)
        export_clicked = action_export.button("Export JSON", use_container_width=True)
        reset_clicked = action_reset.button("Reset", use_container_width=True)

        if reset_clicked:
            reset_app_state()

        if generate_clicked:
            if not base_line.strip():
                st.error("An input line is required.")
            else:
                payload = generate_payload(
                    raw_input=base_line,
                    speaker=speaker,
                    target=target,
                    context=context,
                    input_mode=input_mode,
                    source_block_file=selected_block_file if input_mode == "corpus" else None,
                    source_entry_index=selected_entry_index if input_mode == "corpus" else None,
                )
                dataset_path = append_to_dataset(payload)
                st.session_state.current_payload = payload
                st.success(f"Generated 6 variants and logged this run to {dataset_path}.")

        if export_clicked:
            payload = st.session_state.current_payload
            if not payload:
                st.warning("Generate a result before exporting JSON.")
            else:
                export_path = export_payload(payload)
                st.success(f"Exported JSON to {export_path}.")

    with center_col:
        st.subheader("Tone Variants")
        payload = st.session_state.current_payload
        if not payload:
            st.info("Enter a line and click Generate to create six tone-shaped variants.")
        else:
            st.caption(
                f"Run metadata: version {payload['version']} | "
                f"mode {payload['generation_mode']} | "
                f"strategy {payload['generation_strategy']} | "
                f"input {payload['input_mode']} | "
                f"utterance {payload['utterance_type']} | "
                f"normalized line: {payload['base_line_normalized']}"
            )
            st.write("Normalization")
            st.caption(
                f"Applied: {payload['normalization_applied']} | "
                f"mode: {payload['normalization_mode']} | "
                f"confidence: {payload['normalization_confidence']}"
            )
            st.caption(f"Raw input: {payload['raw_input']}")
            st.caption(f"Generation line: {payload['normalized_generation_input']}")
            if payload["normalization_source_block_file"]:
                st.caption(f"Normalization source block: {payload['normalization_source_block_file']}")
            if payload["normalization_source_text"]:
                st.caption(f"Normalization source text: {payload['normalization_source_text']}")
            st.dataframe(
                build_variant_rows(payload),
                use_container_width=True,
                hide_index=True,
            )
            for index, variant in enumerate(payload["variants"], start=1):
                render_variant_card(index, variant)

    with right_col:
        st.subheader("Structured JSON")
        payload = st.session_state.current_payload
        if not payload:
            st.code(
                '{\n'
                '  "version": "0.1.4",\n'
                '  "generation_mode": "utterance_type_paraphrase_rules_v1",\n'
                '  "generation_strategy": "utterance_type_paraphrase_rules_v1",\n'
                '  "timestamp": "...",\n'
                '  "base_line": "...",\n'
                '  "base_line_normalized": "...",\n'
                '  "input_mode": "manual",\n'
                '  "source_block_file": "",\n'
                '  "source_entry_index": null,\n'
                '  "raw_input": "...",\n'
                '  "normalized_generation_input": "...",\n'
                '  "normalization_applied": false,\n'
                '  "normalization_mode": "none",\n'
                '  "normalization_source_text": "",\n'
                '  "normalization_source_block_file": "",\n'
                '  "normalization_confidence": 1.0,\n'
                '  "utterance_type": "generic_statement",\n'
                '  "context_flags": {},\n'
                '  "variants": []\n'
                '}',
                language="json",
            )
        else:
            st.write("Context flags")
            st.json(payload["context_flags"])
            st.json(payload)
            st.download_button(
                label="Download JSON",
                data=json.dumps(payload, indent=2, ensure_ascii=False),
                file_name="dialogue-tone-workbench-current.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
