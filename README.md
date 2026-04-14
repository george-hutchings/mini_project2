# Confounder-Free Neural Networks for MS Lesion Segmentation

PyTorch implementation of a **confounder-free (CF) U-Net** for multiple sclerosis
lesion segmentation from multi-channel brain MRI. The model follows the
adversarial "bias-resilient" training scheme of
[Zhao et al. (2020)](https://arxiv.org/abs/2002.02561) and
[Dinsdale et al. (2021)](https://doi.org/10.1016/j.neuroimage.2020.117689):
a feature extractor is trained to segment lesions while being explicitly
*decorrelated* from nuisance variables (e.g. scanner site, acquisition
parameters, demographics) via a second network that tries to recover those
confounders from the features.

This repository contains the code that accompanies my MSc mini-project
**"Diffusion Models and Confounder Free Neural Networks"**
(University of Oxford, supervised by Prof. Tom Nichols, Dr. Habib Ganjgahi and
Prof. Chris Holmes). The write-up lives in a companion repository:
[`mini-project-2-report-thesis-style`](https://github.com/george-hutchings/mini-project-2-report-thesis-style).

## Method in one diagram

The U-Net is split into a feature extractor `F` and a 1x1-conv prediction head
`P`. A third network `B` takes the features produced by `F` and predicts the
confounders. The three networks are updated in turn each mini-batch:

| Step | Network | Objective |
|------|---------|-----------|
| 1 | `F` | minimise `L_p - lambda * L_c` (segment well, *decorrelate* from confounders) |
| 2 | `B` | minimise `lambda * L_c` (predict the confounders as well as possible) |
| 3 | `P` | minimise `L_p` (segmentation head) |

`L_p` is Dice + BCE and `L_c = -sum_i corr^2(c_i, c_hat_i)` is the negative
squared Pearson correlation between each true confounder and its prediction.
The update order differs slightly from Zhao et al.; it is equivalent in effect
but avoids a redundant forward pass, which is substantially faster in PyTorch.

## Repository layout

```
.
├── my_unet/                 # model code + training loops + saved checkpoints
│   ├── Models.py            #   F (U-Net encoder/decoder), P (1x1 head), B (confounder predictor)
│   ├── Data_Loader.py       #   torch.utils.data wrappers with on-the-fly augmentation
│   ├── losses.py            #   Dice + BCE segmentation loss and the Pearson-corr^2 loss
│   ├── unet_vanilla_train2.py   # baseline U-Net (no confounder branch)
│   ├── unet_cf_train100_2.py    # confounder-free U-Net (adversarial training)
│   └── model_*/             # trained checkpoints (modelF.pth, modelP.pth, modelB.pth)
└── processing_data/         # data preparation pipeline (run top-to-bottom)
    ├── 1processing_scans/          # normalise + crop MRI volumes (T1, T2, PD, pseudo-FLAIR)
    ├── 2processing_confounders/    # build and standardise the confounder matrix
    ├── 3processing_combine/        # split into train/val/test .npy tensors
    └── 4processing_testdata/       # evaluate a trained model on a held-out cohort
```

Each step under `processing_data/` ships with a SLURM `submit.sh` that was used
on the Oxford BMRC cluster; the Python scripts are self-contained and can be
run directly on any machine with the required data.

## Data

The models expect four-channel 2-D slices stacked into `.npy` tensors:

| Array | Shape | Notes |
|-------|-------|-------|
| `mri_{train,val}.npy`  | `(N, H, W, 4)` | T1, T2, PD, pseudo-FLAIR |
| `mask_{train,val}.npy` | `(N, H, W, 1)` | binary lesion mask |
| `cf_{train,val}.npy`   | `(N, K)`       | `K` standardised confounders |

The raw MRI data cannot be redistributed; the scripts in `processing_data/`
document the exact preprocessing (MAD-based intensity normalisation, centre
cropping to 192x160, brain-mask hole filling, etc.) used to go from raw NIfTI
to these arrays.

## Reproducing a run

```bash
# 1. create an environment
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy scipy scikit-image SimpleITK matplotlib pillow

# 2. point the scripts at the directory holding the .npy tensors
#    (edit `save_dir` near the top of unet_*_train*.py)

# 3. baseline
python my_unet/unet_vanilla_train2.py

# 4. confounder-free
python my_unet/unet_cf_train100_2.py
```

Training writes a timestamped `model_*` directory containing the best
checkpoints for each of `F`, `P` (and `B`, for the CF variant).

## Key implementation details

- **Three decoupled `.backward()` passes per mini-batch** (`my_unet/unet_cf_train100_2.py`)
  — each uses `inputs=` to restrict gradient flow to the relevant submodule,
  avoiding a separate optimiser step per network while still implementing the
  min–max objective.
- **`corr_loss` in `my_unet/losses.py`** is the Pearson correlation squared,
  summed over confounders. Using the *squared* correlation means the adversary
  is indifferent to sign, which matches Zhao et al.'s formulation.
- **Augmentation shares a seed between image and mask** (`Data_Loader.py`) so
  random rotations / crops stay consistent across the input and target.
- **Deterministic seeding** (`torch.manual_seed`, `np.random.seed`,
  `torch.backends.cudnn.deterministic = True`) in both training scripts so runs
  are reproducible.

## References

- Q. Zhao *et al.*, "Training confounder-free deep learning models for medical
  applications," *Nature Communications*, 2020.
- N. K. Dinsdale *et al.*, "Deep learning-based unlearning of dataset bias for
  MRI harmonisation and confound removal," *NeuroImage*, 2021.
- O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for
  Biomedical Image Segmentation," *MICCAI*, 2015.

## Notes for readers

This is research code written under time pressure for a single MSc mini-project
— it is deliberately left close to the form it was run in on the compute
cluster (SLURM submit scripts, cluster-specific paths) rather than reworked
into a package. The purpose of this repository is to show the modelling,
training and evaluation code end-to-end.
