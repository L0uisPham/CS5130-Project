# CheXpert Training Skeleton (Model-Agnostic)

This project provides a unified training/evaluation skeleton for CheXpert so you can plug in models from different sources with the same pipeline.

## Quickstart

Run a single model:
```bash
python -m src.experiments.run_train --config configs/chexpert.yaml --model resnet34 --seed 42
python -m src.experiments.run_train --config configs/chexpert.yaml --model vgg19_hf --seed 42
python -m src.experiments.run_train --config configs/chexpert.yaml --model efficientnet_b4 --seed 42
python -m src.experiments.run_train --config configs/chexpert.yaml --model deit_small --seed 42
python -m src.experiments.run_train --config configs/chexpert.yaml --model swin_tiny --seed 42
python -m src.experiments.run_train --config configs/chexpert.yaml --model convnext_t --seed 42
```

## Results

Each run writes outputs to:
```
runs/{timestamp}_{model_name}_seed{seed}/
```

Inside each run directory:
```
config.yaml
auc_by_epoch.csv
checkpoints/
  best.pt
  last.pt
```

`auc_by_epoch.csv` stores per-epoch AUC rows for:
- each label
- gender (sex=0, sex=1)
- each age bin
