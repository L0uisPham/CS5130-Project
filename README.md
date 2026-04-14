# CS5130 Project

Chest X-ray analysis project for training and comparing deep learning models, serving local model inference through a FastAPI backend, and viewing results in a React/Vite radiology UI.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
pip install -r pipeline/requirements_cpu.txt
cd figma_frontend && npm install
```

## Run

```bash
make run-api
cd figma_frontend && npm run dev
```

To train a model:

```bash
cd pipeline
python -m src.experiments.run_train --config configs/chexpert.yaml --model convnext_t --seed 42
```
