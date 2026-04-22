# Age Estimation from Facial Images with Controlled CNN Architecture Comparisons

This repository contains the code for an age-estimation project on **UTKFace** using PyTorch. The project compares three closely related VGG-style regression models under matched training conditions:

- `vgg`: plain VGG-style baseline
- `resvgg`: VGG-style network with minimal residual connections added
- `unet`: VGG encoder with a lightweight U-Net-style decoder

A standard `resnet50` baseline is also included as a **sanity check**.

The main goal is not only to report which model has the lowest error, but also to analyze **where each architecture works well, where it fails, and how training strategy affects age-specific expertise regions**.

---

## 1. Dataset

This project uses the **UTKFace** dataset.

**Dataset page:**  
`https://susanqq.github.io/UTKFace/`

The age label is read directly from the image filename. Example:

```text
1_0_0_20161219140623097.jpg
```

The first token before the first underscore is the age, so the label for this example is **1**.

### Expected folder layout

Place the dataset in the following structure:

```text
<data_root>/
  part1/   # original train split
  part2/   # original validation split
  part3/   # original test split
```

Example:

```text
project_root/
  part1/
  part2/
  part3/
  train.py
  evaluate.py
  run_experiments.py
  ...
```

### How the current code uses the data

The latest training pipeline does **not** use `part1` and `part2` separately.

Instead, it:

1. pools `part1 + part2`
2. creates a new **90/10 train/validation split**
3. stratifies the split by **exact age** so every observed age remains in the training set
4. keeps `part3` untouched as the final **test set**

The code also writes a `label_counts.csv` file for each run so you can verify age coverage in the pooled, train, and validation sets.

**Important note:** the stored train split is **not physically equalized** across all ages. Instead, the code uses **balanced class sampling during training** so the model sees a flatter age distribution over mini-batches.

---

## 2. Environment setup

### Option A: create the environment from `environment.yml`

```bash
conda env create -f environment.yml
conda activate age-architecture-study
```

### Option B: install into an existing conda environment

At minimum you need:

- Python 3.11+
- PyTorch
- torchvision
- numpy
- pandas
- pillow
- matplotlib
- scikit-learn
- pyyaml
- tqdm

### CUDA / GPU note

If you want GPU training, install a CUDA-enabled build of PyTorch in your conda environment.

To confirm CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

On Windows, it is strongly recommended to launch scripts with:

```bash
python run_experiments.py ...
```

instead of invoking `run_experiments.py` directly, to make sure the currently activated conda environment is used.

---

## 3. Repository structure

```text
README.md
environment.yml
dataset.py               # dataset loading, pooled split creation, age parsing, class-balanced sampler
models.py                # VGG, ResVGG, U-Net-style regressor, ResNet-50 sanity-check baseline
losses.py                # RMSE, MSE, MAE, Huber losses
train.py                 # model training
evaluate.py              # evaluation on part3 test set
analyze_expertise.py     # per-age-bin analysis, expertise regions, plots, hardest cases
gradcam_regression.py    # Grad-CAM visualization for regression models
run_experiments.py       # convenience script to train/evaluate full experiment suites
```

Typical generated outputs:

```text
train_runs/<experiment_name>/
  best.pt
  config.json
  history.csv
  label_counts.csv
  training_summary.json

eval_runs/<experiment_name>/
  predictions.csv
  summary.json

analysis_architectures/
  overall_metrics.csv
  age_bin_metrics.csv
  expertise_regions.csv
  cohort_metrics.csv
  sample_level_winners.csv
  overall_mae.png
  age_bin_bias.png
  age_bin_mae.png
```

---

## 4. Quick start

Run the standard architecture comparison:

```bash
python run_experiments.py --data_root . --output_root . --device auto
```

This will:

1. train the configured models
2. evaluate them on `part3`
3. generate comparison CSV files and plots

If CUDA is available and correctly installed:

```bash
python run_experiments.py --data_root . --output_root . --device cuda
```

If you want to force CPU:

```bash
python run_experiments.py --data_root . --output_root . --device cpu
```

---

## 5. Train a single model

Example: train the best controlled model (`resvgg`) with the current default setup.

```bash
python train.py \
  --data_root . \
  --output_dir train_runs \
  --experiment_name arch_resvgg \
  --model resvgg \
  --pool_splits part1 part2 \
  --val_ratio 0.10 \
  --epochs 20 \
  --batch_size 128 \
  --optimizer adam \
  --lr 3e-4 \
  --loss rmse \
  --sampling balanced_classes \
  --device auto
```

### Key training arguments

- `--model {vgg,resvgg,unet,resnet50}`
- `--epochs 20`
- `--batch_size 128`
- `--optimizer {adam,adamw,sgd}`
- `--lr 3e-4`
- `--loss {rmse,mse,mae,huber}`
- `--sampling {uniform,balanced_classes}`
- `--device {auto,cuda,cpu}`
- `--resnet_pretrained` to use ImageNet-pretrained ResNet-50

---

## 6. Evaluate a trained checkpoint

```bash
python evaluate.py \
  --checkpoint train_runs/arch_resvgg/best.pt \
  --data_root . \
  --test_split part3 \
  --output_dir eval_runs/arch_resvgg \
  --device auto
```

This produces:

- `predictions.csv`: sample-level predictions and errors
- `summary.json`: MAE, RMSE, bias, within-2 / within-5 / within-10, and latency

---

## 7. Run the full architecture comparison

```bash
python run_experiments.py \
  --data_root . \
  --output_root . \
  --device auto \
  --epochs 20 \
  --batch_size 128 \
  --optimizer adam \
  --lr 3e-4 \
  --loss rmse \
  --sampling balanced_classes
```

By default, architecture mode compares:

- `arch_vgg`
- `arch_resvgg`
- `arch_unet`
- `sanity_resnet50`

---

## 8. Run specific ablations

### Loss ablation

```bash
python run_experiments.py --data_root . --output_root . --mode losses --device auto
```

### Learning-rate schedule ablation

```bash
python run_experiments.py --data_root . --output_root . --mode schedules --device auto
```

### Sampling ablation

```bash
python run_experiments.py --data_root . --output_root . --mode sampling --device auto
```

---

## 9. Expertise-region analysis

After evaluation, compare model behavior by age bin:

```bash
python analyze_expertise.py \
  --inputs \
    arch_vgg=eval_runs/arch_vgg/predictions.csv \
    arch_resvgg=eval_runs/arch_resvgg/predictions.csv \
    arch_unet=eval_runs/arch_unet/predictions.csv \
    sanity_resnet50=eval_runs/sanity_resnet50/predictions.csv \
  --output_dir analysis_architectures
```

This generates:

- overall metrics
- per-age-bin metrics
- expertise-region rankings
- cohort metrics
- bias plots
- hardest-error case CSVs

---

## 10. Grad-CAM visualizations

To visualize where a trained model is focusing:

```bash
python gradcam_regression.py \
  --checkpoint train_runs/arch_resvgg/best.pt \
  --predictions_csv eval_runs/arch_resvgg/predictions.csv \
  --select worst \
  --k 10 \
  --output_dir gradcam_outputs \
  --device auto
```

---

## 11. Reproducibility notes

- Seed control is built into training and evaluation scripts.
- `config.json` stores the full run configuration.
- `label_counts.csv` records exact-age coverage after the pooled split is rebuilt.
- `part3` is kept untouched as the final test split.

---

## 12. Dependencies

This project depends mainly on:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `Pillow`
- `matplotlib`
- `scikit-learn`
- `pyyaml`
- `tqdm`

Install them through `environment.yml` or your own conda environment.

---

## 13. Codebase credit and differences from standard implementations

This repository is a **custom project codebase** written for the AI7102 project.

It is **not** a direct copy of an external GitHub implementation. The main custom contributions are:

- a controlled VGG-family comparison instead of unrelated off-the-shelf architectures
- a residualized VGG variant designed as a minimal architectural modification
- a lightweight VGG-based U-Net regressor matched more closely in capacity
- pooled `part1 + part2` splitting with exact-age-aware validation construction
- balanced class sampling for debiasing the training stream
- expertise-region analysis by age bin

The ResNet-50 baseline uses the standard model provided by `torchvision`.

If you use this repository in a report or presentation, also cite the standard architecture papers and the UTKFace dataset page.

---

## 14. Recommended workflow

1. Download and place the UTKFace data into `part1`, `part2`, and `part3`.
2. Create and activate the conda environment.
3. Run the architecture sweep with `run_experiments.py`.
4. Inspect `train_runs/*/label_counts.csv` and `config.json`.
5. Evaluate on `part3`.
6. Run `analyze_expertise.py` to generate tables and plots.
7. Use the outputs in the final report and presentation.

---

## 15. Contact / repo metadata

Add the following before submission:

- team member names
- project report PDF
- private GitHub repo link in the submitted report
- sample output files in the repo if required by your course submission process

