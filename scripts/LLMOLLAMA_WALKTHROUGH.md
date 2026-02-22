# LLMollama.py — Full Code Walkthrough

This document explains every section of `LLMollama.py` and how data flows from x-ray model outputs to a personalized narrative report.

---

## Overview

**Purpose:** Turn **CheXpert-style classifier outputs** (14 labels + probabilities per x-ray) into:
1. **Structured explanation** (JSON: summary, ranked findings, differential, actions, safety note)
2. **Personalized prose report** for that specific x-ray image (2–4 paragraphs)

**Requirements:** [Ollama](https://ollama.ai) running locally with a model (e.g. `llama3.1:8b`). The script does **not** load or process image pixels—it only uses **outputs** from other files (e.g. `inference.py`, `image_processing.py`, or `models/ray.py`) that already produced `labels` and `probs` for an image.

---

## 1) Utility: Probabilities → Status (lines ~36–64)

- **`prob_to_status(p, present_thr=0.70, uncertain_thr=0.30)`**  
  Maps a probability to one of: **"Present"** (≥0.7), **"Uncertain"** (0.3–0.7), **"Not present"** (<0.3). Used so the LLM sees clear categories instead of raw numbers.

- **`build_label_table(labels, probs, present_thr, uncertain_thr)`**  
  Takes the 14 label names and 14 probabilities, clamps probs to [0,1], assigns a status to each, and returns a list of `{label, prob, status}` dicts **sorted by probability descending**. This table is what gets pasted into the first LLM prompt.

---

## 2) Prompt construction for structured JSON (lines ~67–120)

- **`JSON_SCHEMA_EXAMPLE`**  
  Template showing the exact JSON shape the LLM must return: `summary`, `ranked_findings`, `differential`, `recommended_actions`, `safety_note`.

- **`build_prompt(label_table, allowed_labels)`**  
  Builds the **first** prompt:
  - Renders the label table as markdown (Label | Prob | Status).
  - Tells the model to use **only** those labels, no meds, no definitive diagnosis, output **valid JSON only**.
  - Injects the schema example so the model fills it with real content.
  - `allowed_labels` is used to restrict `ranked_findings` to the 14 CheXpert labels (reduces hallucination).

---

## 3) Ollama API call (lines ~124–153)

- **`call_ollama_generate(prompt, model, host, temperature, top_p, timeout_s)`**  
  Sends `POST` to `{host}/api/generate` with the prompt and options. Returns the **raw text** from the model (for the first prompt this should be JSON; for the personalized prompt it’s prose).

---

## 4) JSON parsing and repair (lines ~157–243)

- **`extract_json_candidate(text)`**  
  Finds the first `{` and last `}` in the response and returns that slice. Handles cases where the model adds extra text or markdown around the JSON.

- **`is_forbidden_content(s)`**  
  Simple guardrails: returns True if the string contains medication-like words (e.g. "mg", "prescribe", "antibiotic") or overly definitive language ("definitely", "diagnosis is"). Used to reject unsafe JSON.

- **`validate_output_json(obj, allowed_labels)`**  
  Checks that the parsed object has all required keys, that `ranked_findings` only uses labels from `allowed_labels`, that `safety_note` is non-empty, and that the full JSON doesn’t trigger `is_forbidden_content`. Returns `(ok: bool, reason: str)`.

- **`repair_with_ollama(bad_text, model, host)`**  
  If the first response is invalid JSON or fails validation, this sends a **second** prompt asking the model to fix the output (same schema, no meds, cautious language). The repaired string is then re-parsed and re-validated.

---

## 5) Personalized writing for one x-ray (lines ~247–328)

- **`build_personalized_report_prompt(explanation, image_id)`**  
  Takes the **explanation** dict (from `explain_chexpert_outputs`) and an optional **image_id** (e.g. path or study ID). Builds a **second** prompt that:
  - Pastes summary, ranked findings, differential, recommended actions, and safety note.
  - Optionally says “This report is for the following study/image: …”.
  - Asks for 2–4 paragraphs of **prose only** (no JSON, no bullets): (1) intro that this is model-based for this study, (2) key findings in plain language, (3) differential and next steps, (4) safety disclaimer.

- **`write_personalized_report(explanation=None, labels=None, probs=None, image_id=None, ...)`**  
  **Main entry for personalized text.**
  - If you pass **explanation**, it uses it directly.
  - If you pass **labels + probs** (e.g. from inference), it first calls `explain_chexpert_outputs(labels, probs)` to get the explanation, then builds the personalized prompt and calls Ollama. Returns the narrative string.

---

## 6) Load inputs from file / pipeline (lines ~331–396)

- **`load_inputs_from_file(path)`**  
  Loads JSON from disk. Supports three shapes:
  1. **`{"labels": [...], "probs": [...], "image_id": "..."}`** — raw inference-style output.
  2. **`{"explanation": {...}, "image_id": "..."}`** — pre-computed explanation.
  3. **Top-level explanation** — object that already has `summary`, `ranked_findings`, etc. (optionally with `image_id`).

  Returns a dict that the CLI and `run_personalized_from_file` understand (e.g. `labels`+`probs` or `explanation`, plus `image_id`).

- **`run_personalized_from_file(input_path, output_path=None, model, host)`**  
  Loads from file, calls `write_personalized_report` with the right arguments, and optionally writes the report to **output_path**.

---

## 7) Main structured explanation (lines ~399–353)

- **`explain_chexpert_outputs(labels, probs, model, host, present_thr, uncertain_thr, retries)`**  
  End-to-end for **JSON explanation**:
  1. Build label table → build prompt → call Ollama.
  2. Extract JSON from response; if parse fails, call `repair_with_ollama` and retry (up to `retries`).
  3. Validate with `validate_output_json`; if it fails, ask the model to fix and retry.
  4. Return the validated explanation dict.

---

## 8) CLI (lines ~417–520)

When you run `python scripts/LLMollama.py`:

- **`--mode explain`**  
  Reads `--input` JSON. If it has `labels`+`probs`, runs `explain_chexpert_outputs` and prints (and optionally writes with `--output`) the **JSON explanation**. If the file already contains an explanation, just prints/writes it.

- **`--mode personalize`** (default)  
  Reads `--input` JSON. If it has `labels`+`probs`, runs the full pipeline (explain then personalized report). If it has an explanation, only runs the personalized step. Uses `--image_id` if provided, otherwise `image_id` from the JSON. Prints (and optionally writes with `--output`) the **prose report**.

- **`--input`**  
  Path to JSON (inference output or explanation).

- **`--output`**  
  Optional path to write the result (JSON for explain, .txt for personalize).

- **`--image_id`**  
  Optional override for which image/study the report is for (e.g. path to the x-ray file).

- **`--model`**, **`--host`**  
  Ollama model name and API host.

---

## Data flow summary

```
[Other files: inference.py / image_processing.py / models/ray.py]
        │
        ▼
  labels (14 strings) + probs (14 floats)  [per x-ray image]
        │
        ├──► explain_chexpert_outputs()  ──►  JSON (summary, ranked_findings, differential, actions, safety_note)
        │
        └──► write_personalized_report(labels=..., probs=..., image_id="path/to/xray.png")
                    │
                    ├── (internally calls explain_chexpert_outputs if needed)
                    └──►  Plain-text personalized narrative for that image
```

To test with an example x-ray (no real image file needed for LLM step), use the script in `scripts/run_llmollama_example.py` and the generated `example_xray_input.json`.
