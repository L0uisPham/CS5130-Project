"""
chexpert_llm.py / LLMollama.py

Local LLM explainer for CheXpert-style 14-label outputs using Ollama.

Inputs (from other project files, e.g. inference.py, image_processing.py):
- labels: list[str] length 14 (CheXpert labels)
- probs: list[float] length 14 (0..1)

Outputs (dict from explain_chexpert_outputs):
- summary
- ranked_findings
- differential
- recommended_actions
- safety_note

Personalized writing:
- write_personalized_report() takes the above outputs (or raw labels+probs)
  plus an optional image identifier and generates a short, individualized
  narrative for that x-ray image.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import requests


def _get_dry_run() -> bool:
    """True if LLMOLLAMA_DRY_RUN=1 (or true/yes) to skip Ollama and use mock outputs."""
    return os.environ.get("LLMOLLAMA_DRY_RUN", "").strip().lower() in ("1", "true", "yes")


# -----------------------------
# 1) Utility: probs -> status
# -----------------------------
def prob_to_status(p: float, present_thr: float = 0.70, uncertain_thr: float = 0.30) -> str:
    """
    Convert probability to one of: Present / Uncertain / Not present.
    """
    if p >= present_thr:
        return "Present"
    if p >= uncertain_thr:
        return "Uncertain"
    return "Not present"


def build_label_table(labels: List[str], probs: List[float],
                      present_thr: float = 0.70, uncertain_thr: float = 0.30) -> List[Dict[str, Any]]:
    """
    Build a structured table list from labels and probabilities.
    """
    if len(labels) != len(probs):
        raise ValueError(f"labels and probs must have same length. Got {len(labels)} and {len(probs)}")
    table = []
    for lab, p in zip(labels, probs):
        p_clamped = max(0.0, min(1.0, float(p)))
        table.append({
            "label": lab,
            "prob": round(p_clamped, 4),
            "status": prob_to_status(p_clamped, present_thr, uncertain_thr),
        })
    # Sort by prob descending for readability
    table.sort(key=lambda x: x["prob"], reverse=True)
    return table


# -----------------------------
# 2) Prompt construction
# -----------------------------
JSON_SCHEMA_EXAMPLE = {
    "summary": "A concise 2-3 paragraph synthesis that explains which probabilities matter most, what overall pattern they may suggest, where uncertainty remains, and what kinds of follow-up information or review would be most useful next.",
    "ranked_findings": [
        {"label": "Pleural Effusion", "status": "Present", "prob": 0.91, "rationale": "Short reason tied to the probability/status."}
    ],
    "differential": [
        {"condition": "Possible pneumonia", "confidence": "low/medium/high", "why": "Brief justification using findings."}
    ],
    "recommended_actions": [
        {"action": "Consider clinical correlation and prior imaging comparison", "urgency": "routine/soon/urgent", "note": "Generic, non-prescriptive."}
    ],
    "safety_note": "Research/education use only; not medical advice or a definitive diagnosis."
}


def _format_prompt_buckets(label_table: List[Dict[str, Any]]) -> str:
    """
    Build compact, high-signal guidance from classifier outputs so the LLM can
    synthesize a useful summary instead of merely restating labels.
    """
    present = [row for row in label_table if row["status"] == "Present"]
    uncertain = [row for row in label_table if row["status"] == "Uncertain"]
    not_present = [row for row in label_table if row["status"] == "Not present"]
    very_high = [row for row in label_table if row["prob"] >= 0.85]
    moderate = [row for row in label_table if 0.50 <= row["prob"] < 0.85]
    weak = [row for row in label_table if row["prob"] < 0.30]

    def _fmt(rows: List[Dict[str, Any]], limit: int) -> str:
        if not rows:
            return "None"
        return ", ".join(f"{row['label']} ({row['prob']:.2f})" for row in rows[:limit])

    strongest = label_table[:5]
    near_threshold = [
        row for row in label_table
        if 0.25 <= row["prob"] <= 0.75
    ][:5]
    notable_negatives = not_present[:5]

    return "\n".join([
        f"Strongest model signals: {_fmt(strongest, 5)}",
        f"Very high-confidence signals (>=0.85): {_fmt(very_high, 5)}",
        f"Moderate-strength signals (0.50-0.84): {_fmt(moderate, 5)}",
        f"Present findings (>= threshold): {_fmt(present, 5)}",
        f"Borderline/uncertain findings: {_fmt(uncertain, 5)}",
        f"Notable negatives / low-signal labels: {_fmt(notable_negatives, 5)}",
        f"Near-threshold labels to treat cautiously: {_fmt(near_threshold, 5)}",
        f"Weak/deprioritized signals (<0.30): {_fmt(weak, 5)}",
    ])


def build_prompt(label_table: List[Dict[str, Any]], allowed_labels: List[str]) -> str:
    """
    Build a strict prompt that forces JSON output and limits hallucinations.
    """
    # Compact human-readable table
    lines = ["Label | Prob | Status", "---|---|---"]
    for row in label_table:
        lines.append(f"{row['label']} | {row['prob']:.4f} | {row['status']}")
    table_md = "\n".join(lines)

    allowed_list = ", ".join(allowed_labels)
    prompt_buckets = _format_prompt_buckets(label_table)

    prompt = f"""
You are assisting a research demo that converts chest X-ray classifier outputs into a structured explanation.
Your job is to turn probabilities into a useful next-step interpretation, not just restate label names.
You MUST follow these rules:

RULES:
1) Use ONLY the provided labels and probabilities. Do NOT invent findings not in the label list.
2) Do NOT provide medical advice, prescriptions, or medication recommendations.
3) Do NOT claim certainty or a definitive diagnosis. Use cautious language.
4) Output MUST be valid JSON ONLY. No markdown, no backticks, no extra commentary.
5) In ranked_findings, ONLY use labels from this exact allowed list:
   [{allowed_list}]
6) Provide a short differential (2-4 items) as possibilities, each with low/medium/high confidence.
7) Always include a safety_note stating this is for research/education only.
8) recommended_actions is important. Provide 3-5 specific, non-prescriptive next-step suggestions that are logically tied to the probabilities.
9) Each recommended_actions item must be generic and safe, but still useful. Prefer actions such as:
   - correlate with symptoms, exam findings, oxygenation, or labs
   - compare with prior chest imaging
   - obtain formal radiologist review
   - consider short-interval follow-up or additional workup if clinically warranted
   - flag urgent review only when the strongest probabilities suggest a potentially time-sensitive pattern
10) Set urgency using the probabilities:
   - urgent: reserve for patterns with very high probabilities and potentially serious acute findings
   - soon: use for moderate/high findings that merit timely review or correlation
   - routine: use when outputs are weak, mixed, or mostly uncertain
11) The summary is the main user-facing output. It must be a concise text-only synthesis in 2-3 short paragraphs:
   - paragraph 1: identify the strongest probabilities and explain which findings drive the interpretation
   - paragraph 2: describe the broader pattern they may suggest, and explicitly mention uncertainty or competing interpretations
   - paragraph 3: explain what follow-up context, comparison, or review would be most useful next
12) In ranked_findings rationale, mention the probability strength and why that label matters relative to the others.
13) If no strong positive findings are present, say that clearly, emphasize uncertainty, and avoid forcing a pattern.
14) Keep the summary text-only. Do not use markdown bullets or headings inside the summary.

INPUT (model outputs):
{table_md}

DERIVED CONTEXT (useful synthesis cues, still based only on the table above):
{prompt_buckets}

OUTPUT FORMAT (example schema; fill with actual content):
{json.dumps(JSON_SCHEMA_EXAMPLE, ensure_ascii=False)}

Return JSON only.
""".strip()

    return prompt


# -----------------------------
# 3) Ollama call
# -----------------------------
def _call_ollama_generate_api(
    url: str,
    payload: dict,
    timeout_s: int,
) -> str:
    """POST to Ollama; returns response text. Raises on connection error or non-404 HTTP error."""
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def _call_ollama_chat_api(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    timeout_s: int,
) -> str:
    """Use /api/chat (messages format). Returns message content."""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return msg.get("content", "")


def call_ollama_generate(
    prompt: str,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_s: int = 120,
) -> str:
    """
    Calls Ollama and returns the raw text response.
    Tries POST /api/generate first; if 404, falls back to POST /api/chat (some Windows setups).
    Raises with a clear message if Ollama is not running or both endpoints fail.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    try:
        return _call_ollama_generate_api(url, payload, timeout_s)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not connect to Ollama. Is it running? Start it with:\n  ollama serve\n"
            "Then in another terminal pull and run a model:\n  ollama run llama3.1:8b\n"
            f"Host used: {host}"
        ) from e
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # Fallback: some Windows/Ollama setups only expose /api/chat
            try:
                return _call_ollama_chat_api(host, model, prompt, temperature, top_p, timeout_s)
            except requests.exceptions.HTTPError:
                raise RuntimeError(
                    f"Ollama returned 404 at {url}. Tried /api/chat as fallback and it also failed.\n"
                    "  1) Ensure Ollama is running (ollama serve or open Ollama app).\n"
                    f"  2) Pull the model: ollama run {model}\n"
                    "  3) To test without Ollama: set LLMOLLAMA_DRY_RUN=1 (Windows) or export LLMOLLAMA_DRY_RUN=1 (Mac/Linux)"
                ) from e
        raise
    # Unreachable but satisfy return type
    return ""


# -----------------------------
# 3b) Dry-run: mock explanation without Ollama
# -----------------------------
def build_mock_explanation(
    label_table: List[Dict[str, Any]],
    allowed_labels: List[str],
) -> Dict[str, Any]:
    """
    Build a valid explanation dict from the label table only (no LLM).
    Used when LLMOLLAMA_DRY_RUN=1 so the pipeline can be tested without Ollama.
    """
    present = [r for r in label_table if r["status"] == "Present"]
    uncertain = [r for r in label_table if r["status"] == "Uncertain"]
    negatives = [r for r in label_table if r["status"] == "Not present"]
    if present:
        top = present[:3]
        top_text = ", ".join(f"{r['label']} ({r['prob']:.2f})" for r in top)
        uncertain_text = ", ".join(f"{r['label']} ({r['prob']:.2f})" for r in uncertain[:2])
        negative_text = ", ".join(f"{r['label']} ({r['prob']:.2f})" for r in negatives[:2])
        summary = (
            f"The strongest model signals are {top_text}, and those probabilities should drive the overall interpretation more than the lower-ranked labels.\n\n"
            "Taken together, these findings may point toward a broader chest radiograph pattern rather than a single isolated process, but the outputs remain probabilistic and should be interpreted cautiously.\n\n"
        )
        if uncertain_text:
            summary += f"Additional borderline or mixed-strength signals include {uncertain_text}, which leaves meaningful uncertainty around the exact pattern and whether multiple explanations remain plausible.\n\n"
        if negative_text:
            summary += f"At the same time, lower-signal labels such as {negative_text} are not being prioritized by the model, which helps frame what appears less likely in this output.\n\n"
        summary += "The most useful next step is to interpret these probabilities alongside symptoms, bedside context, oxygenation or lab data when relevant, and any prior chest imaging for comparison."
    else:
        uncertain_text = ", ".join(f"{r['label']} ({r['prob']:.2f})" for r in uncertain[:3])
        summary = "No findings are above the present threshold, so the model is not showing a strong positive signal for a dominant abnormality.\n\n"
        if uncertain_text:
            summary += f"The main signals are borderline or uncertain, including {uncertain_text}, so the output is better read as indeterminate than as a clear impression.\n\n"
        summary += "Low-probability labels remain deprioritized, but that should not be overinterpreted as exclusion. The most useful next step is correlation with the clinical question and comparison with prior imaging or formal review if available."
    ranked_findings = [
        {
            "label": r["label"],
            "status": r["status"],
            "prob": r["prob"],
            "rationale": f"Probability {r['prob']:.2f} maps to {r['status']}.",
        }
        for r in label_table
        if r["label"] in allowed_labels
    ][:10]
    differential = [
        {"condition": "Clinical correlation recommended", "confidence": "low", "why": "Based on model outputs."},
        {"condition": "Consider prior imaging comparison", "confidence": "low", "why": "Routine follow-up."},
    ]
    if present and any(r["prob"] >= 0.85 for r in present):
        recommended_actions = [
            {"action": "Obtain timely clinician or radiologist review of the highest-probability findings", "urgency": "soon", "note": "Escalate concern based on the strongest model signals."},
            {"action": "Correlate the leading findings with symptoms, exam findings, and any available oxygenation or lab data", "urgency": "soon", "note": "Helps determine whether the predicted pattern fits the clinical picture."},
            {"action": "Compare with prior chest imaging if available", "urgency": "routine", "note": "Useful for assessing chronic versus new change."},
        ]
    elif uncertain:
        recommended_actions = [
            {"action": "Compare the uncertain findings with prior chest imaging or formal radiology review", "urgency": "routine", "note": "Helpful when probabilities are mixed or borderline."},
            {"action": "Correlate the leading probabilities with the current clinical question and symptoms", "urgency": "routine", "note": "Supports interpretation without overcalling weak signals."},
            {"action": "Consider follow-up review if the image was obtained for an acute concern and the clinical picture remains unclear", "urgency": "routine", "note": "Non-prescriptive next-step framing."},
        ]
    else:
        recommended_actions = [
            {"action": "Use the output as a low-confidence adjunct rather than a standalone interpretation", "urgency": "routine", "note": "No dominant high-probability finding is present."},
            {"action": "Correlate with the clinical indication and prior studies if available", "urgency": "routine", "note": "Helps contextualize weak signals."},
        ]
    safety_note = "Research/education use only; not medical advice or a definitive diagnosis."
    return {
        "summary": summary,
        "ranked_findings": ranked_findings,
        "differential": differential,
        "recommended_actions": recommended_actions,
        "safety_note": safety_note,
    }


def build_mock_personalized_report(explanation: Dict[str, Any], image_id: Optional[str]) -> str:
    """Prose report template when LLMOLLAMA_DRY_RUN=1 (no Ollama call)."""
    image_line = f" (Study/image: {image_id})" if image_id else ""
    return (
        f"This interpretation is based on model outputs for this specific study{image_line}.\n\n"
        f"{explanation.get('summary', '')}\n\n"
        "Key findings from the model are listed in the structured explanation. "
        "Differential considerations and suggested next steps should be discussed with a qualified provider. "
        "This is for research and education only; it is not medical advice or a definitive diagnosis."
    )


# -----------------------------
# 4) JSON parsing + repair
# -----------------------------
def extract_json_candidate(text: str) -> str:
    """
    Tries to extract a JSON object from a messy response.
    If it's already clean JSON, returns as-is.
    """
    text = text.strip()
    # If the model accidentally adds leading/trailing text, attempt to slice to first {...last}
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return text


def is_forbidden_content(s: str) -> bool:
    """
    Lightweight guardrails to keep output safe-ish.
    """
    lower = s.lower()
    # Medication / prescription hints
    meds = ["mg", "tablet", "dose", "prescribe", "antibiotic", "ibuprofen", "acetaminophen", "tylenol", "advil"]
    if any(m in lower for m in meds):
        return True
    # Overly definitive diagnosis language
    definite = ["definitely", "confirmed", "certain", "this is", "diagnosis is", "will have"]
    if any(d in lower for d in definite):
        return True
    return False


def validate_output_json(obj: Dict[str, Any], allowed_labels: List[str]) -> Tuple[bool, str]:
    """
    Validate schema and hallucination constraints.
    Returns (ok, reason).
    """
    required_top = ["summary", "ranked_findings", "differential", "recommended_actions", "safety_note"]
    for k in required_top:
        if k not in obj:
            return False, f"Missing top-level key: {k}"

    if not isinstance(obj["ranked_findings"], list):
        return False, "ranked_findings must be a list"
    for item in obj["ranked_findings"]:
        if not isinstance(item, dict):
            return False, "ranked_findings items must be objects"
        if "label" not in item:
            return False, "ranked_findings item missing label"
        if item["label"] not in allowed_labels:
            return False, f"Hallucinated label in ranked_findings: {item['label']}"

    # Safety note must exist and be non-trivial
    if not isinstance(obj["safety_note"], str) or len(obj["safety_note"].strip()) < 10:
        return False, "safety_note too short or invalid"

    # Guardrails check: scan stringified JSON for forbidden patterns
    if is_forbidden_content(json.dumps(obj).lower()):
        return False, "Forbidden/unsafe content detected (meds or definitive diagnosis language)"

    return True, "OK"


def repair_with_ollama(
    bad_text: str,
    model: str,
    host: str,
    timeout_s: int = 120,
) -> str:
    """
    Ask model to fix JSON only.
    """
    fix_prompt = f"""
You returned invalid or non-compliant output. Fix it.

RULES:
- Return ONLY valid JSON (no markdown/backticks).
- Preserve the same schema keys: summary, ranked_findings, differential, recommended_actions, safety_note.
- Remove any medication/prescription content.
- Avoid definitive diagnosis language. Use cautious language.
- Do not add any labels that were not already used.

BAD OUTPUT:
{bad_text}

Return corrected JSON only.
""".strip()
    return call_ollama_generate(fix_prompt, model=model, host=host, temperature=0.1, top_p=0.9, timeout_s=timeout_s)


# -----------------------------
# 5) Personalized writing for an individualized x-ray image
# -----------------------------
def build_personalized_report_prompt(
    explanation: Dict[str, Any],
    image_id: Optional[str] = None,
) -> str:
    """
    Build a prompt that asks the LLM to write a short, personalized narrative
    for a single x-ray image based on the structured explanation.
    """
    summary = explanation.get("summary", "No summary provided.")
    ranked = explanation.get("ranked_findings", [])
    differential = explanation.get("differential", [])
    actions = explanation.get("recommended_actions", [])
    safety = explanation.get("safety_note", "")

    lines = [f"Summary: {summary}", "", "Ranked findings:"]
    for r in ranked[:10]:
        lines.append(f"  - {r.get('label', '')}: {r.get('status', '')} (prob {r.get('prob', 0):.2f}) — {r.get('rationale', '')}")
    lines.append("")
    lines.append("Differential considerations:")
    for d in differential:
        lines.append(f"  - {d.get('condition', '')} ({d.get('confidence', '')}): {d.get('why', '')}")
    lines.append("")
    lines.append("Recommended actions:")
    for a in actions:
        lines.append(f"  - [{a.get('urgency', '')}] {a.get('action', '')} — {a.get('note', '')}")
    lines.append("")
    lines.append(f"Safety note: {safety}")

    context = "\n".join(lines)
    image_line = f"\nThis report is for the following study/image: {image_id}." if image_id else ""

    prompt = f"""
You are helping produce a short, personalized radiology-style narrative for one chest x-ray study.
Use ONLY the structured findings below. Do not invent findings. Do not give medication or prescriptions.
Use cautious language (e.g. "suggests", "consider", "possible"). Do not state a definitive diagnosis.

STRUCTURED FINDINGS FOR THIS IMAGE:
{context}
{image_line}

Write 2–4 short paragraphs that:
1) Open with a sentence that this interpretation is based on model outputs for this specific study.
2) Summarize the key findings in plain language, ordered by relevance.
3) Briefly mention differential considerations and suggested next steps.
4) End with the safety disclaimer (research/education only; not medical advice).

Output ONLY the narrative text. No JSON, no markdown headers, no bullet lists—just prose.
""".strip()
    return prompt


def write_personalized_report(
    explanation: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    probs: Optional[List[float]] = None,
    image_id: Optional[str] = None,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    temperature: float = 0.3,
    timeout_s: int = 120,
) -> str:
    """
    Generate a personalized narrative for an individualized x-ray image.

    Inputs (use one of):
    - explanation: dict from explain_chexpert_outputs() (summary, ranked_findings, etc.)
    - labels + probs: raw CheXpert outputs from inference (e.g. from inference.py or image_processing pipeline);
      will call explain_chexpert_outputs() internally to get the explanation first.

    - image_id: optional identifier for this image (e.g. path, study ID) so the narrative is clearly for "this" image.

    Returns:
    - Plain-text personalized report string.
    """
    if explanation is None:
        if labels is None or probs is None:
            raise ValueError("Provide either explanation dict or both labels and probs.")
        explanation = explain_chexpert_outputs(labels, probs, model=model, host=host)

    if _get_dry_run():
        return build_mock_personalized_report(explanation, image_id)

    prompt = build_personalized_report_prompt(explanation, image_id=image_id)
    raw = call_ollama_generate(prompt, model=model, host=host, temperature=temperature, timeout_s=timeout_s)
    return raw.strip()


# -----------------------------
# 6) Load inputs from other files / JSON (pipeline integration)
# -----------------------------
def load_inputs_from_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load inputs from a JSON file produced by other scripts (e.g. inference or a wrapper).

    Expected JSON shapes (any of):
    - {"labels": [...], "probs": [...]}  → use with explain_chexpert_outputs or write_personalized_report
    - {"explanation": {...}, "image_id": "..."}  → use explanation + image_id for personalized report
    - {"summary": ..., "ranked_findings": ..., ...}  → full explanation dict at top level
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If it's already a full explanation (has summary + ranked_findings), return as-is for "explanation" key
    if "summary" in data and "ranked_findings" in data:
        return {"explanation": data, "image_id": data.get("image_id")}

    if "explanation" in data:
        return {"explanation": data["explanation"], "image_id": data.get("image_id")}

    if "labels" in data and "probs" in data:
        return {"labels": data["labels"], "probs": data["probs"], "image_id": data.get("image_id")}

    raise ValueError(
        "JSON must contain either (labels, probs) or (explanation) or top-level (summary, ranked_findings). "
        f"Keys found: {list(data.keys())}"
    )


def run_personalized_from_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
) -> str:
    """
    Load inputs from a JSON file (output of other pipeline files), generate personalized
    report for the x-ray image, and optionally write it to a file.

    Returns the report string.
    """
    loaded = load_inputs_from_file(input_path)
    image_id = loaded.get("image_id")

    if "explanation" in loaded:
        report = write_personalized_report(
            explanation=loaded["explanation"],
            image_id=image_id,
            model=model,
            host=host,
        )
    else:
        report = write_personalized_report(
            labels=loaded["labels"],
            probs=loaded["probs"],
            image_id=image_id,
            model=model,
            host=host,
        )

    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return report


# -----------------------------
# 7) Main structured explanation function
# -----------------------------
def explain_chexpert_outputs(
    labels: List[str],
    probs: List[float],
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    present_thr: float = 0.70,
    uncertain_thr: float = 0.30,
    retries: int = 1,
) -> Dict[str, Any]:
    """
    End-to-end: label probs -> prompt -> Ollama -> validated JSON dict.

    Raises RuntimeError if it can't produce valid output after retries.
    If LLMOLLAMA_DRY_RUN=1, returns a mock explanation without calling Ollama.
    """
    label_table = build_label_table(labels, probs, present_thr, uncertain_thr)

    if _get_dry_run():
        return build_mock_explanation(label_table, allowed_labels=labels)

    prompt = build_prompt(label_table, allowed_labels=labels)
    raw = call_ollama_generate(prompt, model=model, host=host)
    candidate = extract_json_candidate(raw)

    for attempt in range(retries + 1):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            if attempt >= retries:
                raise RuntimeError(f"LLM output not valid JSON after {retries} retries.\nRaw:\n{raw}")
            candidate = repair_with_ollama(candidate, model=model, host=host)
            candidate = extract_json_candidate(candidate)
            continue

        ok, reason = validate_output_json(obj, allowed_labels=labels)
        if ok:
            return obj

        if attempt >= retries:
            raise RuntimeError(f"LLM JSON failed validation: {reason}\nJSON:\n{json.dumps(obj, indent=2)}")

        # Ask model to fix compliance issues
        candidate = repair_with_ollama(json.dumps(obj), model=model, host=host)
        candidate = extract_json_candidate(candidate)

    raise RuntimeError("Unreachable: retries loop ended unexpectedly.")


# -----------------------------
# 8) CLI: run from pipeline with outputs from other files
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM explainer and personalized report writer for CheXpert x-ray outputs (Ollama)."
    )
    parser.add_argument(
        "--mode",
        choices=["explain", "personalize"],
        default="personalize",
        help="explain: output JSON only. personalize: write personalized narrative for the image (default).",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON with labels+probs or explanation (+ optional image_id). From inference or other scripts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write result to this file (explain → JSON; personalize → text).",
    )
    parser.add_argument(
        "--image_id",
        type=str,
        default=None,
        help="Override or set image/study ID for personalized report (e.g. path to x-ray image).",
    )
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama API host.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Ollama; use mock explanation and report (set LLMOLLAMA_DRY_RUN=1).",
    )
    args = parser.parse_args()

    if args.dry_run:
        os.environ["LLMOLLAMA_DRY_RUN"] = "1"

    if args.input:
        loaded = load_inputs_from_file(args.input)
        image_id = args.image_id or loaded.get("image_id")
        if args.mode == "explain":
            if "explanation" in loaded:
                result = json.dumps(loaded["explanation"], indent=2)
            else:
                obj = explain_chexpert_outputs(
                    loaded["labels"], loaded["probs"], model=args.model, host=args.host
                )
                result = json.dumps(obj, indent=2)
            if args.output:
                Path(args.output).write_text(result, encoding="utf-8")
            print(result)
        else:
            if "explanation" in loaded:
                report = write_personalized_report(
                    explanation=loaded["explanation"],
                    image_id=image_id,
                    model=args.model,
                    host=args.host,
                )
            else:
                report = write_personalized_report(
                    labels=loaded["labels"],
                    probs=loaded["probs"],
                    image_id=image_id,
                    model=args.model,
                    host=args.host,
                )
            if args.output:
                Path(args.output).write_text(report, encoding="utf-8")
            print(report)
    else:
        parser.print_help()
        print("\nExample: save labels+probs from inference, then run:")
        print('  python scripts/LLMollama.py --input inference_output.json --output report.txt --image_id "path/to/xray.png"')
        print("  python scripts/LLMollama.py --input explanation.json --mode personalize --output report.txt")
        print("\nWithout Ollama (mock output):  python scripts/LLMollama.py --input scripts/example_xray_input.json --dry-run --output report.txt")
