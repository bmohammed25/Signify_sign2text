# Signify — Sign-to-Text (LSTM Landmark Model)

ASL word-level recognition using MediaPipe landmark sequences.  
Trained on ASL Citizen + WLASL across **2,410 classes**.

> **Fanshawe College · Group 7 · Summer 2026**

---

## What this is

Signify_sign2text trains a lightweight LSTM classifier (`SignifyLSTM`) on
MediaPipe body-landmark sequences extracted from ASL video. Each video is
reduced to a `(30, 1629)` NumPy array — 30 frames × 543 landmarks × 3 coords
(face 468 + pose 33 + left hand 21 + right hand 21). No video frames, no CNNs,
no heavy GPU needed: a Quadro T1000 with 4 GB VRAM is sufficient.

**Baseline to beat:** 53.67% test top-1 / 60.2% val top-1  
(previous run, 46,806 train videos, epoch 81)

**This run:** 73,838 train videos (80/10/10 re-split), all 2,410 classes,
cleaner contiguous labels — targeting a meaningful improvement over that baseline.

---

## Model — SignifyLSTM

```
Input  (B, 30, 1629)
  └─ LayerNorm(1629)
  └─ Linear(1629 → 512) + ReLU          # feature projection
  └─ LSTM(512, hidden=512, layers=3, dropout=0.3)
  └─ h_n[-1]  →  (B, 512)               # last hidden state, top layer
  └─ BN → Dropout(0.4) → Linear(512, 256) → ReLU
  └─ BN → Dropout(0.3) → Linear(256, 2410)
Output (B, 2410)   raw logits
```

~7.89 M trainable parameters.

---

## Repository layout

```
Signify_sign2text/
├── notebooks/
│   └── train_signify_lstm.ipynb   # full training pipeline, cell-by-cell
├── src/
│   └── model_b.py                 # SignifyLSTM architecture
├── data/
│   ├── processed/
│   │   ├── train.csv              # 73,838 rows  (80 %)
│   │   ├── val.csv                #  9,533 rows  (10 %)
│   │   └── test.csv               #  9,533 rows  (10 %)
│   └── landmarks/                 # 92,904 × (30,1629) .npy — NOT in git
│                                  # download from Hugging Face (see below)
├── checkpoints/                   # saved during training — NOT in git
│   ├── last.pt                    # latest epoch (for resume)
│   └── best.pt                    # best val top-1 → uploaded to HF
├── requirements.txt
└── README.md
```

`data/landmarks/` and `checkpoints/` are gitignored.  
Large files live on Hugging Face (links below).

---

## Dataset & weights (Hugging Face)

| Resource | Link |
|---|---|
| Landmark dataset (`.npy` files) | _add HF dataset link here_ |
| Trained weights (`best.pt`) | _add HF model link here_ |

---

## Data sources

| Source | Videos | Notes |
|---|---|---|
| ASL Citizen | ~65 K | Primary source |
| WLASL | ~27 K | Supplementary; some videos missing landmarks |

Combined, re-split 80 / 10 / 10. Labels remapped to contiguous `class_idx`
0 … 2409.

---

## Environment

| Item | Value |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| OS tested | Windows 11 |
| GPU tested | Quadro T1000 (4 GB) |

Install dependencies:

```bash
pip install -r requirements.txt
```

Or restore the full conda environment:

```bash
conda env create -f environment.yml   # if provided
conda activate signify
```

---

## Training

Open `notebooks/train_signify_lstm.ipynb` and run cells top to bottom.  
All hyperparameters are defined in **Cell 1** (Step 1) — one place, named
variables, referenced everywhere. Key defaults:

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR (T_max = 100) |
| Grad clip | 1.0 |
| Early stopping | patience = 15 epochs |
| `num_workers` | 0 (Windows) |

Training saves `checkpoints/last.pt` every epoch and `checkpoints/best.pt`
whenever val top-1 improves. Resume is supported — the checkpoint stores
epoch, model, optimizer, and scheduler state.

---

## Results

| Run | Train videos | Test top-1 | Val top-1 | Epoch |
|---|---|---|---|---|
| Previous best (old split) | 46,806 | 53.67 % | 60.2 % | 81 |
| This run | 73,838 | — | — | — |

_Table will be updated after training completes._

---

## License

Code: MIT.  
ASL Citizen and WLASL datasets are subject to their own licenses — see the
respective dataset pages before redistribution.
