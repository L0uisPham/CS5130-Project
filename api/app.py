"""
FastAPI server that runs local checkpoint inference + LLMollama and returns JSON
in the shape expected by the React Chest X-ray AI Analysis app.

Start (from project root):
  pip install -r api/requirements.txt
  python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8001

Then in the React app, call POST http://localhost:8001/analyze with the image
file and model_name form field.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
import logging

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
PIPELINE_ROOT = ROOT / "pipeline"
if str(PIPELINE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PIPELINE_ROOT))

WEIGHTS_DIR = ROOT / "weights"

LOCAL_MODEL_OPTIONS = [
    {
        "value": "convnext_t",
        "label": "ConvNeXt Tiny",
        "weightsPath": str(WEIGHTS_DIR / "convnext.pt"),
        "builder": "torchvision_convnext_tiny",
    },
    {
        "value": "swin_tiny",
        "label": "Swin Tiny",
        "weightsPath": str(WEIGHTS_DIR / "swin_best.pt"),
        "hfName": "microsoft/swin-tiny-patch4-window7-224",
        "builder": "hf_image_classification",
    },
    {
        "value": "ensemble",
        "label": "ConvNeXt + Swin Ensemble",
        "builder": "ensemble",
    },
]
DEFAULT_LOCAL_MODEL = LOCAL_MODEL_OPTIONS[0]["value"]
logger = logging.getLogger(__name__)

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
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8001",
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

    raw_csv_lines = ["label,probability,status"]
    for item in label_list:
        raw_csv_lines.append(
            f'{item["label"]},{item["probability"]:.4f},{item["status"]}'
        )

    return {
        "labels": label_list,
        "llm": {
            "summary": explanation.get("summary", ""),
            "rankedFindings": ranked_findings,
            "differentials": differentials,
            "recommendedActions": recommended_actions,
            "safetyNote": explanation.get("safety_note", ""),
        },
        "rawCsv": "\n".join(raw_csv_lines),
    }


def _labels_and_probs_to_frontend(labels: list, probs: list) -> dict:
    """
    Map classifier outputs only, without waiting for the LLM explanation.
    """
    return _pipeline_output_to_frontend(
        labels,
        probs,
        explanation={
            "summary": "",
            "ranked_findings": [],
            "differential": [],
            "recommended_actions": [],
            "safety_note": "",
        },
    )


def _explanation_to_frontend(explanation: dict) -> dict:
    """
    Map explanation-only output into the frontend LLM block shape.
    """
    payload = _pipeline_output_to_frontend([], [], explanation)
    return payload["llm"]


def _load_pipeline_cfg() -> dict:
    import yaml

    chexpert_cfg_path = PIPELINE_ROOT / "configs" / "chexpert.yaml"
    with chexpert_cfg_path.open("r", encoding="utf-8") as fh:
        chexpert_cfg = yaml.safe_load(fh) or {}
    chexpert_cfg["num_classes"] = len(chexpert_cfg.get("labels", []))
    return chexpert_cfg


def _get_model_option(model_name: str) -> dict:
    for option in LOCAL_MODEL_OPTIONS:
        if option["value"] == model_name:
            return option
    raise HTTPException(400, f"Unknown local model: {model_name}")


def _extract_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint format is unsupported.")

    for key in ("state_dict", "model_state_dict", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def _build_local_model(model_option: dict, num_classes: int):
    builder = model_option.get("builder", "timm")

    if builder == "torchvision_convnext_tiny":
        from torch import nn
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if builder == "timm":
        from timm import create_model

        return create_model(
            model_option["arch"],
            pretrained=False,
            num_classes=num_classes,
        )

    if builder == "hf_image_classification":
        try:
            from transformers import AutoConfig, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load this model. Install api dependencies again."
            ) from exc

        hf_name = model_option["hfName"]
        config = AutoConfig.from_pretrained(hf_name)
        config.num_labels = num_classes
        return AutoModelForImageClassification.from_config(config)

    raise RuntimeError(f"Unsupported local model builder: {builder}")


@lru_cache(maxsize=1)
def _get_ensemble_predictor():
    from src.ensemble.ensemble import Ensemble

    return Ensemble(
        config_path=PIPELINE_ROOT / "configs" / "chexpert.yaml",
        weights_dir=WEIGHTS_DIR,
    )


def _predict_with_local_model(image_path: str, model_name: str) -> tuple[list[str], list[float]]:
    import torch
    from PIL import Image
    from torchvision import transforms

    cfg = _load_pipeline_cfg()
    labels = list(cfg.get("labels", []))
    if not labels:
        raise RuntimeError("CheXpert config is missing labels.")

    model_option = _get_model_option(model_name)
    if model_option.get("builder") == "ensemble":
        ensemble = _get_ensemble_predictor()
        return ensemble.inference_from_path(image_path)

    weights_path = Path(model_option["weightsPath"])
    if not weights_path.exists():
        raise RuntimeError(f"Weights file not found: {weights_path}")

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    x = eval_tfms(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = model_option.get("builder", "timm")
    model = _build_local_model(model_option, num_classes=len(labels)).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} parameters for {model_name}: "
            + ", ".join(missing[:5])
        )
    if unexpected:
        raise RuntimeError(
            f"Checkpoint has unexpected parameters for {model_name}: "
            + ", ".join(unexpected[:5])
        )
    model.eval()
    with torch.no_grad():
        if builder == "hf_image_classification":
            logits = model(pixel_values=x.to(device)).logits
        else:
            logits = model(x.to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return labels, probs


def _run_pipeline_then_llm(image_path: str, model_name: str, dry_run: bool) -> dict:
    import os as _os

    from scripts.LLMollama import (
        build_label_table,
        build_mock_explanation,
        build_mock_personalized_report,
        explain_chexpert_outputs,
        write_personalized_report,
    )

    labels, probs = _predict_with_local_model(image_path, model_name)
    if dry_run:
        _os.environ["LLMOLLAMA_DRY_RUN"] = "1"

    try:
        explanation = explain_chexpert_outputs(labels, probs)
        report = write_personalized_report(
            explanation=explanation,
            image_id=image_path,
        )
    except RuntimeError as exc:
        logger.warning(
            "LLM step failed for model %s. Falling back to deterministic explanation. %s",
            model_name,
            exc,
        )
        label_table = build_label_table(labels, probs)
        explanation = build_mock_explanation(label_table, allowed_labels=labels)
        report = build_mock_personalized_report(explanation, image_id=image_path)

    return {
        "labels": labels,
        "probs": probs,
        "explanation": explanation,
        "report": report,
    }


def _run_llm_only(labels: list[str], probs: list[float], dry_run: bool) -> dict:
    import os as _os

    from scripts.LLMollama import (
        build_label_table,
        build_mock_explanation,
        explain_chexpert_outputs,
    )

    if dry_run:
        _os.environ["LLMOLLAMA_DRY_RUN"] = "1"

    try:
        return explain_chexpert_outputs(labels, probs)
    except RuntimeError as exc:
        logger.warning("LLM step failed. Falling back to deterministic explanation. %s", exc)
        label_table = build_label_table(labels, probs)
        return build_mock_explanation(label_table, allowed_labels=labels)


def _validate_uploaded_image(image_path: str) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            400,
            "The uploaded file is not a supported image or is corrupted.",
        ) from e


@app.get("/health")
def health():
    return {"status": "ok", "message": "Chest X-ray AI API is running."}


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "value": option["value"],
                "label": option["label"],
            }
            for option in LOCAL_MODEL_OPTIONS
        ],
        "defaultModel": DEFAULT_LOCAL_MODEL,
    }


def _mock_response(model_name: str = DEFAULT_LOCAL_MODEL) -> dict:
    """Return static mock ModelOutput for API_MOCK=1 or when pipeline fails (e.g. no weights)."""
    return {
        "modelUsed": model_name,
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
            "summary": "The strongest model signals are pleural effusion, cardiomegaly, and lung opacity, with those findings standing above the remaining labels and therefore driving the overall interpretation.\n\nTaken together, this combination may reflect a broader cardiopulmonary or fluid-overload pattern rather than a single isolated abnormality, although the output remains probabilistic and should be interpreted cautiously.\n\nThere is still meaningful uncertainty in the surrounding picture because enlarged cardiomediastinum, edema, consolidation, and pneumonia remain in the borderline range, while lower-signal labels such as pneumothorax, fracture, and lung lesion are not being prioritized by the model.\n\nThese results are most useful when correlated with symptoms, prior imaging, and formal clinical interpretation. This analysis is for research and education only and is not medical advice.",
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
        "rawCsv": "\n".join([
            "label,probability,status",
            "Pleural Effusion,0.9200,present",
            "Cardiomegaly,0.7800,present",
            "Lung Opacity,0.7100,present",
            "Enlarged Cardiomediastinum,0.6800,uncertain",
            "Edema,0.6500,uncertain",
            "Consolidation,0.5400,uncertain",
            "Pneumonia,0.4800,uncertain",
            "Atelectasis,0.4200,uncertain",
            "Pleural Other,0.2300,not-present",
            "Pneumothorax,0.1500,not-present",
            "Lung Lesion,0.1200,not-present",
            "Fracture,0.0800,not-present",
            "No Finding,0.0500,not-present",
            "Support Devices,0.0300,not-present",
        ]),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_LOCAL_MODEL),
):
    """
    Upload a chest x-ray image. Runs classifier inference and returns label
    probabilities immediately so the UI can render before the LLM finishes.
    Set API_MOCK=1 to always return mock data (no model or LLM run).
    """
    valid_model_names = {item["value"] for item in LOCAL_MODEL_OPTIONS}
    if model_name not in valid_model_names:
        raise HTTPException(400, f"Unknown local model: {model_name}")

    if os.environ.get("API_MOCK", "").strip().lower() in ("1", "true", "yes"):
        await file.read()  # consume upload
        mock = _mock_response(model_name)
        return {
            "modelUsed": model_name,
            "labels": mock["labels"],
            "rawCsv": mock["rawCsv"],
        }

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
        _validate_uploaded_image(tmp_path)
        labels, probs = _predict_with_local_model(tmp_path, model_name=model_name)
    except ImportError as e:
        logger.exception("Pipeline dependencies are missing. Falling back to mock response.")
        mock = _mock_response(model_name)
        return {
            "modelUsed": model_name,
            "labels": mock["labels"],
            "rawCsv": mock["rawCsv"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze pipeline failed for model %s. Falling back to mock response.", model_name)
        mock = _mock_response(model_name)
        return {
            "modelUsed": model_name,
            "labels": mock["labels"],
            "rawCsv": mock["rawCsv"],
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    frontend_payload = _labels_and_probs_to_frontend(labels, probs)
    frontend_payload["modelUsed"] = model_name
    return frontend_payload


@app.post("/explain")
async def explain(
    payload: dict = Body(...),
):
    """
    Generate the LLM explanation from already-computed classifier outputs.
    This lets the frontend render probabilities first and fill in the LLM panel later.
    """
    labels = payload.get("labels")
    probs = payload.get("probs")

    if not isinstance(labels, list) or not isinstance(probs, list):
        raise HTTPException(400, "Payload must include labels and probs arrays.")
    if len(labels) != len(probs):
        raise HTTPException(400, "labels and probs must have the same length.")

    if os.environ.get("API_MOCK", "").strip().lower() in ("1", "true", "yes"):
        return _mock_response().get("llm", {})

    try:
        dry_run = os.environ.get("LLMOLLAMA_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
        explanation = _run_llm_only(labels, probs, dry_run=dry_run)
    except ImportError:
        logger.exception("LLM dependencies are missing. Falling back to mock explanation.")
        return _mock_response().get("llm", {})
    except Exception:
        logger.exception("Explain step failed. Falling back to mock explanation.")
        return _mock_response().get("llm", {})

    return _explanation_to_frontend(explanation)
