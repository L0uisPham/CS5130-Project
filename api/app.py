"""
FastAPI server that runs the GitHub models + LLMollama pipeline and returns
JSON in the shape expected by the Figma/React Chest X-ray AI Analysis app.

Start (from project root):
  pip install fastapi uvicorn python-multipart
  python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Then in the React app, call:  POST http://localhost:8000/analyze  with the image file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

app = FastAPI(
    title="Chest X-ray AI Analysis API",
    description="Runs vision model + LLMollama pipeline; returns data for Figma/React UI.",
)

# Allow React (Vite default 5173, Next 3000, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prob_to_status(p: float, present_thr: float = 0.70, uncertain_thr: float = 0.30) -> str:
    """Return frontend-style status: present | uncertain | not-present."""
    if p >= present_thr:
        return "present"
    if p >= uncertain_thr:
        return "uncertain"
    return "not-present"


def _pipeline_output_to_frontend(labels: list, probs: list, explanation: dict) -> dict:
    """
    Map pipeline output (labels, probs, explanation) to the React ModelOutput shape.
    """
    # Labels: { label, probability, status }[]
    label_list = []
    for lab, prob in zip(labels, probs):
        p = max(0.0, min(1.0, float(prob)))
        label_list.append({
            "label": lab,
            "probability": round(p, 4),
            "status": _prob_to_status(p),
        })

    # Sort by probability descending (optional, frontend may sort too)
    label_list.sort(key=lambda x: x["probability"], reverse=True)

    # LLM block: summary, rankedFindings, differentials, recommendedActions, safetyNote
    ranked = explanation.get("ranked_findings", [])
    ranked_findings = []
    for r in ranked:
        status = (r.get("status") or "Not present").lower().replace(" ", "-")
        if status == "not-present":
            status = "not-present"
        elif status == "uncertain":
            status = "uncertain"
        else:
            status = "present"
        ranked_findings.append({
            "label": r.get("label", ""),
            "status": status,
            "probability": round(float(r.get("prob", 0)), 4),
            "rationale": r.get("rationale", ""),
        })

    differential = explanation.get("differential", [])
    differentials = []
    for d in differential:
        conf = (d.get("confidence") or "low").lower()
        differentials.append({
            "condition": d.get("condition", ""),
            "confidence": conf if conf in ("low", "medium", "high") else "low",
            "reason": d.get("why", d.get("reason", "")),
        })

    actions = explanation.get("recommended_actions", [])
    recommended_actions = []
    for a in actions:
        u = (a.get("urgency") or "routine").lower()
        recommended_actions.append({
            "action": a.get("action", ""),
            "urgency": u if u in ("routine", "soon", "urgent") else "routine",
        })

    return {
        "labels": label_list,
        "llm": {
            "summary": explanation.get("summary", ""),
            "rankedFindings": ranked_findings,
            "differentials": differentials,
            "recommendedActions": recommended_actions,
            "safetyNote": explanation.get("safety_note", ""),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "Chest X-ray AI API is running."}


def _mock_response() -> dict:
    """Return static mock ModelOutput for API_MOCK=1 or when pipeline fails (e.g. no weights)."""
    return {
        "labels": [
            {"label": "Pleural Effusion", "probability": 0.92, "status": "present"},
            {"label": "Cardiomegaly", "probability": 0.78, "status": "present"},
            {"label": "Lung Opacity", "probability": 0.71, "status": "present"},
            {"label": "Enlarged Cardiomediastinum", "probability": 0.68, "status": "uncertain"},
            {"label": "Edema", "probability": 0.65, "status": "uncertain"},
            {"label": "Consolidation", "probability": 0.54, "status": "uncertain"},
            {"label": "Pneumonia", "probability": 0.48, "status": "uncertain"},
            {"label": "Atelectasis", "probability": 0.42, "status": "uncertain"},
            {"label": "Pleural Other", "probability": 0.23, "status": "not-present"},
            {"label": "Lung Lesion", "probability": 0.12, "status": "not-present"},
            {"label": "Pneumothorax", "probability": 0.15, "status": "not-present"},
            {"label": "Fracture", "probability": 0.08, "status": "not-present"},
            {"label": "Support Devices", "probability": 0.03, "status": "not-present"},
            {"label": "No Finding", "probability": 0.05, "status": "not-present"},
        ],
        "llm": {
            "summary": "The chest X-ray demonstrates findings consistent with moderate pleural effusion and cardiomegaly. This analysis is for research/education only.",
            "rankedFindings": [
                {"label": "Pleural Effusion", "status": "present", "probability": 0.92, "rationale": "High probability from model output."},
                {"label": "Cardiomegaly", "status": "present", "probability": 0.78, "rationale": "Enlarged cardiac silhouette suggested by model."},
            ],
            "differentials": [
                {"condition": "Clinical correlation recommended", "confidence": "medium", "reason": "Based on model outputs."},
            ],
            "recommendedActions": [
                {"action": "Compare with prior imaging if available", "urgency": "soon"},
                {"action": "Obtain formal radiologist interpretation", "urgency": "soon"},
            ],
            "safetyNote": "This analysis is generated by AI for educational and research purposes only. Not medical advice.",
        },
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a chest x-ray image. Runs the pipeline (model + LLMollama) and returns
    JSON in the shape expected by the React app (ModelOutput).
    Set API_MOCK=1 to always return mock data (no model or LLM run).
    """
    if os.environ.get("API_MOCK", "").strip().lower() in ("1", "true", "yes"):
        await file.read()  # consume upload
        return _mock_response()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (e.g. image/png, image/jpeg)")

    suffix = Path(file.filename or "image").suffix or ".png"
    if suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        suffix = ".png"

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}") from e

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from scripts.connect_models_to_llmollama import run_model_then_llm

        # Use ray backend by default; set env API_BACKEND=sk / API_MODEL=resnet to override
        backend = os.environ.get("API_BACKEND", "ray").strip().lower()
        model_name = os.environ.get("API_MODEL", "resnet34").strip()
        dry_run = os.environ.get("LLMOLLAMA_DRY_RUN", "").strip().lower() in ("1", "true", "yes")

        result = run_model_then_llm(
            tmp_path,
            backend=backend,
            model_name=model_name,
            weights_dir=ROOT / "models",
            explanation_only=False,
            report_only=False,
            dry_run=dry_run,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            503,
            "Model or data not found. Train models (e.g. python models/ray.py --mode train) or set API_MOCK=1 to use mock data.",
        ) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    explanation = result.get("explanation")
    if not explanation:
        raise HTTPException(500, "Pipeline did not return an explanation.")

    labels = result.get("labels", [])
    probs = result.get("probs", [])
    return _pipeline_output_to_frontend(labels, probs, explanation)
