# CIFAR-10: MLPs, CNNs, Autoencoders, SVMs and Clustering

**Ashwani Sah** — Roll No. M26MAC007 — IIT Jodhpur
Machine Learning, Assignment 3

## Repository contents

| Path | Contents |
|---|---|
| `Assignment.ipynb` | All code for Q1–Q4 (data split, models, training, evaluation, figures) |
| `splits/` | Saved indices: `train_idx.npy`, `val_idx.npy` (stratified 45k/5k split), `q1b_noisy_idx.npy` (Q1(b) noisy labels), `q2b_pixel_perm.npy` (Q2(b) pixel permutation), `q4_subset_idx.npy` (Q4 10k SVM subset) |
| `configs/config.json` | All hyperparameters and selected values |
| `logs/` | Final per-epoch histories and result tables (`all_results.json`, `all_results.pkl`) |
| `Figure/` | All figures used in the report |
| `environment.yml` | Software environment with exact versions |

## Setup

- Seed: `26007` (all digits of the roll number, in order), used for the train/validation split.
- Final models are trained with seeds `26007`, `26008`, `26009`; results are mean ± std.
- The official CIFAR-10 test set is used only once, after all choices are fixed on validation.
- Hardware: Apple MacBook M4 Pro, PyTorch on the MPS (Apple GPU) backend.
  Results on MPS can differ very slightly between runs even with fixed seeds.

## Reproduce

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate cifar10-a3

# 2. Run all experiments (CIFAR-10 downloads automatically to ./data)
#    Full run takes about 5-6 hours on an M4 Pro (most of it is the CNN training in Q2).
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 Assignment.ipynb
```

The notebook recreates the split from the seed; the saved indices in `splits/`
can be used to verify that the split is identical.

## Selected settings

| Part | Selected (on validation) |
|---|---|
| Q1(a) MLPs | SGD lr 0.01, dropout p = 0.2, 40 epochs |
| Q1(b) | MLP 3072-2048-2048-10, 40% label noise, 150 epochs |
| Q2 CNN | SGD lr 0.03, 40 epochs |
| Q3(a) | AE-512 (bottleneck 512) |
| Q3(b) | Gaussian noise σ = 0.15, DAE-512 |
| Q4(a) | Linear SVM, C = 0.001, on 256-d CNN features (10k subset) |
| Q4(c) | k-means, k = 10, on PCA-50 of the same features |

## Main test results

| Experiment | Test result |
|---|---|
| Q1(a) MLP-base / wide / dropout / shift | 56.17 / 57.51 / 58.23 / 59.73 % |
| Q2 CNN vs MLP (clean) | 84.78 % vs 56.17 % |
| Q2 CNN vs MLP (pixel-shuffled) | 46.64 % vs 55.81 % |
| Q2 CNN vs MLP (patch-shuffled) | 59.46 % vs 30.95 % |
| Q3(a) AE-512 reconstruction | MSE 0.00214 (PSNR 26.71 dB) |
| Q4(a) selected SVM | 84.87 % |
| Q4(c) k-means | ARI 0.669, NMI 0.696 |

Not included: the CIFAR-10 data (downloaded automatically) and trained model
weights (`*.pt`, large; reproducible from the notebook and seeds).
