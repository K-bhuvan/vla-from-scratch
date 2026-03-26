#!/bin/bash
set -e

# vla-from-scratch quick start
# Usage: ./quick_start.sh [sft|posttrain|eval|all]

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"

# Setup environment
echo "Setting up conda environment..."
if ! conda env list | grep -q "^vla "; then
    conda env create -f environment.yml -n vla
fi
conda activate vla

# Run stage
STAGE=${1:-all}

case $STAGE in
    sft)
        echo "Stage B: Supervised Fine-Tuning..."
        python src/train/sft.py --config configs/sft.yaml
        ;;
    posttrain)
        echo "Stage C: Post-training..."
        python src/posttrain/dagger.py --config configs/posttrain.yaml
        ;;
    eval)
        echo "Stage D: Evaluation..."
        python src/eval/evaluate.py --config configs/eval.yaml
        ;;
    all)
        echo "Running all available stages..."
        python src/train/sft.py --config configs/sft.yaml
        python src/posttrain/dagger.py --config configs/posttrain.yaml
        python src/eval/evaluate.py --config configs/eval.yaml
        ;;
    *)
        echo "Usage: ./quick_start.sh [sft|posttrain|eval|all]"
        exit 1
        ;;
esac

echo "Done!"
