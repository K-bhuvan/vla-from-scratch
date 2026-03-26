# Step 1: Project Setup and Environment

## Goal
Establish a clean, reproducible local environment for the VLA project and verify that the GPU stack works before writing model code.

## What was created
- `environment.yml`
- `.gitignore`
- base repository structure under `src/`, `data/`, `docs/`, `configs/`

## Why this step matters
If the environment is unstable, every later step becomes noisy and hard to debug. For this project, the GPU setup mattered immediately because the RTX 5060 Ti is Blackwell architecture and older PyTorch builds did not include compatible kernels.

## Key outcome
The project now uses a dedicated `vla` conda environment, and a quick Python check confirms:
- Python imports work
- PyTorch can see CUDA
- GPU kernels execute correctly
- required packages are installed

## Important machine-specific note
This machine required PyTorch nightly with CUDA 12.8 wheels for Blackwell support. That is why the `vla` environment does not use the older stable CUDA 12.1 setup.

## How to run
```bash
conda activate vla
cd <repo-root>
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Expected result
You should see a successful environment check, including:
- GPU name
- VRAM total
- GPU matmul smoke test

## What you learned
- why environment setup is part of ML engineering, not an afterthought
- why GPU architecture and CUDA compatibility matter
- why each project should have its own isolated environment

## Output of this step
A working local foundation for all future VLA training steps.
