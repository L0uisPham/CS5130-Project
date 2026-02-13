# Package Installation:
pip install -r requirements.txt

# Split the dataset 
python -m src.datasets.splitters

# Run commands:
python -m src.experiments.run_train --config configs/chexpert.yaml --model resnet34 --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model vgg19_hf --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model efficientnet_b4 --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model deit_small --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model swin_tiny --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model convnext_t --seed 42 --batch_size 32 --num_workers 4 --prefetch_factor 2

python -m src.experiments.run_train --config configs/chexpert.yaml --model convnext_t --seed 42 --from_scratch
