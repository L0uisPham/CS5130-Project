"""
Test run for LLMollama.py with an example x-ray (simulated inference output).

This script:
1. Creates example_xray_input.json with CheXpert labels + fake probs (simulating
   output from inference.py or your image pipeline for one x-ray).
2. Runs LLMollama in two modes: --mode explain (JSON) and --mode personalize (prose).
3. Optionally uses an image path as image_id if you have a real x-ray file.

Prerequisites for real LLM output:
- Ollama installed and running (e.g. ollama serve, then ollama run llama3.1:8b).

To test WITHOUT Ollama (mock output):
- Run:  python scripts/run_llmollama_example.py --dry-run
- Or:   set LLMOLLAMA_DRY_RUN=1  then  python scripts/run_llmollama_example.py

From project root: python scripts/run_llmollama_example.py [--dry-run]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# CheXpert 14 labels (same order as in LLMollama / inference / models.ray)
CHEXPERT_LABELS_14 = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def make_example_probs() -> list[float]:
    """Fake probabilities for one 'example' x-ray: Cardiomegaly and Pleural Effusion elevated."""
    # Index order matches CHEXPERT_LABELS_14
    return [
        0.05,   # No Finding - low
        0.25,   # Enlarged Cardiomediastinum
        0.88,   # Cardiomegaly - high (simulate finding)
        0.45,   # Lung Opacity
        0.12,   # Lung Lesion
        0.35,   # Edema
        0.22,   # Consolidation
        0.18,   # Pneumonia
        0.40,   # Atelectasis
        0.08,   # Pneumothorax
        0.79,   # Pleural Effusion - high (simulate finding)
        0.15,   # Pleural Other
        0.05,   # Fracture
        0.62,   # Support Devices - moderate
    ]


def main() -> None:
    if "--dry-run" in sys.argv:
        os.environ["LLMOLLAMA_DRY_RUN"] = "1"
        sys.argv.remove("--dry-run")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    example_json = script_dir / "example_xray_input.json"
    example_image_id = "example_study_001.png"  # or use a real path, e.g. "data/processed_chexpert/images/patient_001.jpg"

    # 1) Write example inference output (what inference.py or your pipeline would save)
    payload = {
        "labels": CHEXPERT_LABELS_14,
        "probs": make_example_probs(),
        "image_id": example_image_id,
    }
    with open(example_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {example_json} (simulated inference output for one x-ray).")
    print()

    # 2) Run LLMollama: explain mode (structured JSON)
    llmollama = project_root / "scripts" / "LLMollama.py"
    explain_out = script_dir / "example_explanation.json"
    dry_run = os.environ.get("LLMOLLAMA_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    print("--- Step 1: Explain (structured JSON)" + (" [DRY RUN - no Ollama]" if dry_run else "") + " ---")
    cmd_explain = [
        sys.executable,
        str(llmollama),
        "--input", str(example_json),
        "--mode", "explain",
        "--output", str(explain_out),
    ]
    if dry_run:
        cmd_explain.append("--dry-run")
    r1 = subprocess.run(cmd_explain, cwd=str(project_root))
    if r1.returncode != 0:
        print("Explain step failed. Is Ollama running? Try: ollama serve then ollama run llama3.1:8b")
        print("Or run without Ollama: python scripts/run_llmollama_example.py --dry-run")
        sys.exit(r1.returncode)
    print(f"Saved explanation to {explain_out}")
    print()

    # 3) Run LLMollama: personalize mode (prose report for this x-ray)
    report_out = script_dir / "example_personalized_report.txt"
    print("--- Step 2: Personalized report (prose for this x-ray)" + (" [DRY RUN - no Ollama]" if dry_run else "") + " ---")
    cmd_personalize = [
        sys.executable,
        str(llmollama),
        "--input", str(example_json),
        "--mode", "personalize",
        "--output", str(report_out),
        "--image_id", example_image_id,
    ]
    if dry_run:
        cmd_personalize.append("--dry-run")
    r2 = subprocess.run(cmd_personalize, cwd=str(project_root))
    if r2.returncode != 0:
        print("Personalize step failed.")
        sys.exit(r2.returncode)
    print(f"Saved report to {report_out}")
    print()
    print("Done. To use a real x-ray image path as image_id, run:")
    print('  python scripts/LLMollama.py --input scripts/example_xray_input.json --output my_report.txt --image_id "C:/path/to/your/xray.png"')


if __name__ == "__main__":
    main()
