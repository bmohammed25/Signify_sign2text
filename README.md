# 🤟 Signify_sign2text

> Real-time American Sign Language recognition powered by deep learning.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Notebooks vs Production Files](#notebooks-vs-production-files)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Running the Notebooks](#running-the-notebooks)
- [Running the App](#running-the-app)
- [Team](#team)

---

## Project Overview

Signify is a real-time ASL (American Sign Language) hand sign recognition application built as a Fanshawe College Capstone project. The system uses a fine-tuned MobileNetV2 convolutional neural network to classify 29 ASL classes (A–Z, space, delete, nothing) from live webcam input.

The project pipeline covers dataset exploration, preprocessing, model training, evaluation, and a live inference application — all documented step-by-step in numbered Jupyter notebooks.

---

## Model Performance

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Dataset | ASL Alphabet (Kaggle) |
| Classes | 29 (A–Z + space, delete, nothing) |
| Accuracy | **95.65%** |
| Framework | PyTorch 2.5.1+cu121 |

---

## Repository Structure

```
signify/
│
├── notebooks/                  # Jupyter notebooks — training & exploration (numbered 01–08)
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocess_data.ipynb
│   ├── 03_build_model.ipynb
│   ├── 04_train_model.ipynb
│   ├── 05_evaluate_model.ipynb
│   ├── 06_confusion_matrix.ipynb
│   ├── 07_test_inference.ipynb
│   └── 08_export_model.ipynb
│
├── src/                        # Production .py files — live app logic
│   ├── config.py               # Paths, constants, model config
│   ├── create_necessary_folders.py        # Creates folder structure on first run
│   ├── verify_env.py           # Checks environment dependencies
│   ├── preprocess.py           # Data transforms (reusable functions)
│   ├── model.py                # MobileNetV2 model definition
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   └── app.py                  # Live webcam inference app
│
├── data/                       # Dataset directory (not committed to Git)
│   └── asl_alphabet/
│       ├── asl_alphabet_train/
│       └── asl_alphabet_test/
│
├── models/                     # Saved model weights (not committed to Git)
│   └── signify_mobilenetv2.pth
│
├── requirements.txt            # Python dependencies
├── verify_env.py               # Quick environment check (run this first)
├── setup_project.py            # Project folder initializer
└── README.md
```

---

## Notebooks vs Production Files

This project separates **exploration/training** (notebooks) from **live application** (.py files). Understanding this distinction is important for working with the codebase.

### Jupyter Notebooks (`notebooks/`)

Notebooks are for **training, experimentation, and documentation**. They are meant to be run once (or a few times) in order, and they produce artifacts like trained model weights and evaluation charts.

- Run sequentially: `01` → `02` → `03` → ... → `08`
- Each notebook is self-contained with markdown explanations
- Outputs include trained `.pth` model files, confusion matrices, and accuracy plots
- **Not** used in the live app — they are the build pipeline

### Production Files (`src/`)

`.py` files are for the **live inference application**. They import cleaned, reusable functions extracted from the notebooks.

- `config.py` — machine-agnostic paths using `os.path.abspath`
- `app.py` — the real-time webcam app (run this to use Signify)
- `verify_env.py` / `setup_project.py` — setup utilities

> **In short:** Use notebooks to train. Use `.py` files to run the app.

---

## Dataset

Signify uses the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

### Download Instructions

1. Install the Kaggle CLI (already included in `requirements.txt`):
   ```bash
   pip install kaggle
   ```

2. Place your Kaggle API key at `~/.kaggle/kaggle.json`.
   Get it from: [https://www.kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings)

3. Download and extract the dataset:
   ```bash
   kaggle datasets download -d grassknoted/asl-alphabet
   unzip asl-alphabet.zip -d data/asl_alphabet/
   ```

4. Your `data/` folder should look like:
   ```
   data/
   └── asl_alphabet/
       ├── asl_alphabet_train/   ← 87,000 training images
       └── asl_alphabet_test/    ← 29 test images
   ```

> The `data/` and `models/` directories are listed in `.gitignore` and are **not committed to the repository**.

---

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Python 3.10
- NVIDIA GPU with CUDA 12.1 support (recommended; CPU fallback available)
- Git

### Step-by-step Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-org/signify.git
cd signify
```

**2. Create the conda environment**
```bash
conda create -n signify python=3.10 -y
conda activate signify
```

**3. Install PyTorch with CUDA support**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**4. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

**5. Verify your environment**
```bash
python verify_env.py
```

Expected output:
```
✅ Python       3.10.x
✅ PyTorch      2.5.1+cu121
✅ CUDA         Available — NVIDIA Quadro T1000
✅ OpenCV       4.x.x
✅ All checks passed. Environment is ready.
```

**6. Initialize the project folder structure**
```bash
python setup_project.py
```

---

## Running the Notebooks

Run notebooks **in order**. Each notebook builds on the outputs of the previous one.

> Activate your conda environment before launching Jupyter:
> ```bash
> conda activate signify
> jupyter notebook
> ```

| # | Notebook | Description |
|---|---|---|
| 01 | `01_explore_dataset.ipynb` | Load and visualize the ASL dataset |
| 02 | `02_preprocess_data.ipynb` | Resize, normalize, augment images |
| 03 | `03_build_model.ipynb` | Define MobileNetV2 architecture |
| 04 | `04_train_model.ipynb` | Train the model, save weights |
| 05 | `05_evaluate_model.ipynb` | Accuracy, loss curves |
| 06 | `06_confusion_matrix.ipynb` | Per-class confusion matrix |
| 07 | `07_test_inference.ipynb` | Test on static images |
| 08 | `08_export_model.ipynb` | Export final model for the app |

---

## Running the App

After completing all notebooks and exporting the model:

```bash
conda activate signify
python src/app.py
```

A webcam window will open. Hold an ASL hand sign in front of the camera and Signify will display the predicted letter in real time.

Press `Q` to quit.

---

## Team

**Fanshawe College — School of Information Technology**
AIM Program — Capstone 2026, Group 7

---

*Built with PyTorch, OpenCV, and a lot of hand signs.*
