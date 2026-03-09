"""
Run Ollama (LLMollama) using the arrays output from the repo vision models directly.
No hypothetical or mock probabilities — only the real labels and probs from the model.

Flow:
  1. Load image and run the chosen vision model (ray or sk) → get labels[], probs[]
  2. Pass those arrays directly to LLMollama (Ollama) for explanation and report

Usage (from project root):

  # Use ray model (resnet34 or effb4); requires models/resnet34_chexpert.pt etc.
  python scripts/run_ollama_with_model_arrays.py --image path/to/xray.png --backend ray --model resnet34

  # Use sk model (resnet, deit, swin, vgg, efficient); requires sk/tuned_models/
  python scripts/run_ollama_with_model_arrays.py --image path/to/xray.png --backend sk --model resnet

  # Save outputs
  python scripts/run_ollama_with_model_arrays.py --image path/to/xray.png --output report.txt --explanation-out explanation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vision model on image, then pass its label/prob arrays to Ollama (no hypothetical probs)."
    )
    parser.add_argument("--image", required=True, help="Path to chest x-ray image.")
    parser.add_argument("--backend", choices=["ray", "sk"], default="ray", help="Vision model: ray (models/ray.py) or sk (sk/inference).")
    parser.add_argument("--model", default="resnet34", help="ray: resnet34, effb4. sk: resnet, deit, swin, vgg, efficient.")
    parser.add_argument("--weights-dir", default=None, help="Weights directory for ray (default: project models/).")
    parser.add_argument("--output", "-o", default=None, help="Write personalized report to this file.")
    parser.add_argument("--explanation-out", default=None, help="Write explanation JSON to this file.")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--llm-host", default="http://localhost:11434", help="Ollama API host.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        print("Use a real path to your x-ray file (e.g. --image \"C:\\Users\\Ray\\Desktop\\my_xray.png\").", file=sys.stderr)
        sys.exit(1)

    # Do not use dry_run: we want real model arrays and real Ollama
    from scripts.connect_models_to_llmollama import run_model_then_llm

    weights_dir = Path(args.weights_dir) if args.weights_dir else ROOT / "models"
    print("Running vision model on image to get label/prob arrays...")
    result = run_model_then_llm(
        str(image_path),
        backend=args.backend,
        model_name=args.model,
        weights_dir=weights_dir,
        explanation_only=False,
        report_only=False,
        dry_run=False,  # always use real Ollama with model arrays
        llm_model=args.llm_model,
        llm_host=args.llm_host,
    )

    labels = result["labels"]
    probs = result["probs"]
    explanation = result["explanation"]
    report = result["report"]

    print("\n--- Model output (arrays passed to Ollama) ---")
    for lab, p in zip(labels, probs):
        print(f"  {lab}: {p:.4f}")
    print()

    if args.explanation_out:
        with open(args.explanation_out, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2)
        print(f"Saved explanation to {args.explanation_out}")

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Saved report to {args.output}")

    print("\n--- Personalized report (from Ollama using model arrays) ---\n")
    print(report)


if __name__ == "__main__":
    main()
