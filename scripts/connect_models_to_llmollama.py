"""
Connect repo models (scripts/sk/inference or models/ray) to LLMollama.

Runs a CheXpert-style model on an x-ray image to get (labels, probs), then feeds
that into LLMollama to produce the structured explanation and personalized report.

Usage (from project root):

  # Use a model trained by models/ray.py (ResNet34 or EfficientNet-B4)
  python scripts/connect_models_to_llmollama.py --image path/to/xray.png --backend ray --model resnet34 --output report.txt

  # Use a model from scripts/sk/tuned_models (deit, swin, resnet, vgg, efficient)
  python scripts/connect_models_to_llmollama.py --image path/to/xray.png --backend sk --model resnet --output report.txt

  # Save explanation JSON as well
  python scripts/connect_models_to_llmollama.py --image path/to/xray.png --backend ray --model resnet34 --output report.txt --explanation-out explanation.json

Use from another script:

  from scripts.connect_models_to_llmollama import run_model_then_llm
  out = run_model_then_llm("path/to/xray.png", backend="ray", model_name="resnet34")
  print(out["report"])
  # out["explanation"] is the structured JSON; out["labels"], out["probs"] are the model outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_llmollama():
    """Load LLMollama module from scripts/ without requiring a package."""
    spec = importlib.util.spec_from_file_location(
        "llmollama", ROOT / "scripts" / "LLMollama.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llmollama"] = mod
    spec.loader.exec_module(mod)
    return mod


def predict_with_ray(image_path: str, model_name: str = "resnet34", weights_dir: Path = None) -> Tuple[List[str], List[float]]:
    """Run models/ray.py model (ResNet34 or EffB4) on one image. Returns (labels, probs)."""
    from models.ray import predict_single_image
    weights_dir = weights_dir or ROOT / "models"
    return predict_single_image(image_path, model_name=model_name, weights_dir=weights_dir)


def predict_with_sk(image_path: str, model_name: str) -> Tuple[List[str], List[float]]:
    """Run scripts/sk/inference model (deit, swin, resnet, vgg, efficient) on one image. Returns (labels, probs)."""
    from scripts.sk.inference.inference import Inference
    inf = Inference(model_name)
    return inf.inference_from_path(image_path)


def run_model_then_llm(
    image_path: str,
    backend: str = "ray",
    model_name: str = "resnet34",
    weights_dir: Path = None,
    explanation_only: bool = False,
    report_only: bool = False,
    llm_model: str = "llama3.1:8b",
    llm_host: str = "http://localhost:11434",
    dry_run: bool = False,
) -> dict:
    """
    Run the chosen classifier on the image, then LLMollama to get explanation and/or report.

    backend: "ray" (models/ray.py) or "sk" (scripts/sk/inference)
    model_name: for ray use "resnet34" or "effb4"; for sk use "resnet", "deit", "swin", "vgg", "efficient"
    Returns: {"labels": [...], "probs": [...], "explanation": {...} or None, "report": str or None}
    """
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if backend.strip().lower() == "ray":
        labels, probs = predict_with_ray(image_path, model_name=model_name, weights_dir=weights_dir)
    elif backend.strip().lower() == "sk":
        labels, probs = predict_with_sk(image_path, model_name=model_name)
    else:
        raise ValueError(f"backend must be 'ray' or 'sk'. Got: {backend}")

    # Ensure we have 14 for LLMollama (ray always returns 14; sk may return 14 or fewer)
    if len(labels) != 14 or len(probs) != 14:
        from models.ray import CHEXPERT_LABELS_14
        target_labels = list(CHEXPERT_LABELS_14)
        if len(probs) >= 14:
            probs = probs[:14]
            labels = labels[:14]
        else:
            probs = list(probs) + [0.0] * (14 - len(probs))
            labels = target_labels[:14]

    llm = _load_llmollama()
    if dry_run:
        import os
        os.environ["LLMOLLAMA_DRY_RUN"] = "1"

    explanation = None
    report = None

    if not report_only:
        explanation = llm.explain_chexpert_outputs(
            labels, probs, model=llm_model, host=llm_host
        )
    if not explanation_only:
        report = llm.write_personalized_report(
            labels=labels,
            probs=probs,
            image_id=image_path,
            model=llm_model,
            host=llm_host,
        )

    return {
        "labels": labels,
        "probs": probs,
        "explanation": explanation,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run repo model on an x-ray image and feed results into LLMollama for explanation and report."
    )
    parser.add_argument("--image", required=True, help="Path to chest x-ray image.")
    parser.add_argument("--backend", choices=["ray", "sk"], default="ray",
                        help="Model source: ray (models/ray.py) or sk (scripts/sk/inference).")
    parser.add_argument("--model", default="resnet34",
                        help="For ray: resnet34, effb4. For sk: resnet, deit, swin, vgg, efficient.")
    parser.add_argument("--weights-dir", default=None,
                        help="Directory with .pt weights (ray backend). Default: project models/")
    parser.add_argument("--output", "-o", default=None,
                        help="Write personalized report to this file.")
    parser.add_argument("--explanation-out", default=None,
                        help="Write explanation JSON to this file.")
    parser.add_argument("--explanation-only", action="store_true",
                        help="Only produce structured explanation (no prose report).")
    parser.add_argument("--report-only", action="store_true",
                        help="Only produce prose report (skip saving explanation JSON).")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--llm-host", default="http://localhost:11434", help="Ollama API host.")
    parser.add_argument("--dry-run", action="store_true", help="Skip Ollama; use mock LLM output.")
    args = parser.parse_args()

    result = run_model_then_llm(
        args.image,
        backend=args.backend,
        model_name=args.model,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
        explanation_only=args.explanation_only,
        report_only=args.report_only,
        llm_model=args.llm_model,
        llm_host=args.llm_host,
        dry_run=args.dry_run,
    )

    if result["explanation"] is not None:
        import json
        exp = result["explanation"]
        if args.explanation_out:
            Path(args.explanation_out).write_text(json.dumps(exp, indent=2), encoding="utf-8")
            print(f"Wrote explanation to {args.explanation_out}")
        if not args.report_only:
            print("Explanation summary:", exp.get("summary", "")[:200] + "...")

    if result["report"] is not None:
        if args.output:
            Path(args.output).write_text(result["report"], encoding="utf-8")
            print(f"Wrote report to {args.output}")
        print("\n--- Report ---\n")
        print(result["report"])


if __name__ == "__main__":
    main()
