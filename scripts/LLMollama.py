"""
chexpert_llm.py

Local LLM explainer for CheXpert-style 14-label outputs using Ollama.

Inputs:
- labels: list[str] length 14
- probs: list[float] length 14 (0..1)

Outputs (dict):
- summary
- ranked_findings
- differential
- recommended_actions
- safety_note
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Tuple, Optional

import requests


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
    "summary": "1-3 sentence findings summary based ONLY on the labels/probabilities.",
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

    prompt = f"""
You are assisting a research demo that converts chest X-ray classifier outputs into a structured explanation.
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

INPUT (model outputs):
{table_md}

OUTPUT FORMAT (example schema; fill with actual content):
{json.dumps(JSON_SCHEMA_EXAMPLE, ensure_ascii=False)}

Return JSON only.
""".strip()

    return prompt


# -----------------------------
# 3) Ollama call
# -----------------------------
def call_ollama_generate(
    prompt: str,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_s: int = 120,
) -> str:
    """
    Calls Ollama /api/generate and returns the raw text response (should be JSON string).
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
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


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
# 5) Main function you call
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
    """
    label_table = build_label_table(labels, probs, present_thr, uncertain_thr)
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