# Data Directory

Raw dataset files are **never committed** to this repository.
Use the scripts below (added in Baby Step 2) to generate or download data.

---

## Dataset Provenance Table

| Dataset | Source | License | How we use it | Download / Generate |
|---------|--------|---------|---------------|---------------------|
| Simulation demos — reach_target | Self-generated (pure NumPy oracle) | N/A — fully synthetic | Pre-training, SFT, post-training | `python data/generate_sim_data.py` |
| Simulation demos — pick_object (primitive) | Self-generated (pure NumPy oracle) | N/A — fully synthetic | SFT, post-training | `python data/generate_sim_data.py --tasks pick_object` |
| Simulation demos — place_object (primitive) | Self-generated (pure NumPy oracle) | N/A — fully synthetic | SFT, post-training | `python data/generate_sim_data.py --tasks place_object` |
| Simulation demos — pick_and_place_object (composed) | Self-generated (pure NumPy oracle) | N/A — fully synthetic | SFT, post-training | `python data/generate_sim_data.py --tasks pick_and_place_object` |

> This table must be updated every time a new data source is added.
> Never use a dataset without adding an entry here first.

---

## Folder Layout (after data generation)

```
data/
├── README.md                  ← this file
├── generate_sim_data.py       ← added in Baby Step 2
├── raw/                       ← gitignored; generated locally
│   └── sim_demos/
│       ├── reach_target/
│       ├── pick_object/
│       ├── place_object/
│       └── pick_and_place_object/
└── processed/                 ← gitignored; preprocessed tensors
    ├── pretrain/
    └── sft/
```

The `raw/` and `processed/` folders are in `.gitignore`.
Only scripts and metadata are committed.
