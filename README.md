<<<<<<< HEAD
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

=======
# 🤟 Signify — ASL Sign Language Communication App

> Real-time American Sign Language recognition powered by deep learning.
> Open Source Project

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Notebooks vs Production Files](#notebooks-vs-production-files)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [Team](#team)

---

## Project Overview

Signify is a real-time ASL (American Sign Language) hand sign recognition application built as a Fanshawe College Capstone project. The system uses a fine-tuned MobileNetV2 convolutional neural network to classify 29 ASL classes (A–Z, space, delete, nothing) from live webcam input. Recognized letters are assembled into natural sentences using a local LLM (phi3:mini via Ollama) and spoken aloud via text-to-speech.

The project pipeline covers dataset exploration, preprocessing, model training, evaluation, and a live inference application — all documented step-by-step in numbered Jupyter notebooks.

---

## Model Performance

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Dataset | ASL Alphabet (Kaggle) |
| Classes | 29 (A–Z + space, delete, nothing) |
| Validation Accuracy | **99.83%** (best at epoch 19) |
| Framework | PyTorch 2.5.1+cu121 |
| GPU | NVIDIA Quadro T1000 4GB |

---

## Repository Structure

```
Signify/
│
├── Notebooks/                  # Jupyter notebooks — training & exploration (numbered 01–07)
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocess.ipynb         ⚠️ Early landmark-based approach (not final pipeline)
│   ├── 03_augment.ipynb            ⚠️ Early landmark-based approach (not final pipeline)
│   ├── 04_model.ipynb              ⚠️ Early ANN approach (not final pipeline)
│   ├── 05_train_mobilenet.ipynb    ✅ Final model training
│   ├── 06_evaluate.ipynb           ✅ Model evaluation
│   └── 07_camera_test.ipynb        ✅ Live camera inference test
│
├── Agents/                     # Production agent pipeline
│   ├── __init__.py
│   ├── vision_agent.py         # Hand detection & crop (MediaPipe)
│   ├── landmark_agent.py       # Quality gate
│   ├── recognition_agent.py    # ASL letter classification (MobileNetV2)
│   ├── language_agent.py       # Sentence assembly (phi3:mini via Ollama)
│   └── speech_agent.py         # Text to speech (pyttsx3)
│
├── API/                        # FastAPI server
│   ├── __init__.py
│   └── main.py                 # All endpoints — run this to start the server
│
├── front-end/                  # Web interface
│   └── signify_app_auto_with_video.html
│
├── Saved Models/               # Trained model weights
│   └── best_model_mobilenet.pth
│
├── Environment/                # Environment setup documentation
│   └── Local_Environment_Setup.docx
│
├── config.py                   # Machine-agnostic paths and constants
├── create_necessary_folders.py # Creates project folder structure on first run
├── verify_env.py               # Checks all dependencies are installed
└── requirements.txt            # Python dependencies
```

---

## Notebooks vs Production Files

This project separates **exploration/training** (notebooks) from **live application** (agents + API).

### Jupyter Notebooks (`Notebooks/`)

Notebooks are for **training, experimentation, and documentation**. Run them sequentially to reproduce the training pipeline.

| # | Notebook | Description |
|---|---|---|
| 01 | `01_explore_dataset.ipynb` | Load and visualize the ASL dataset |
| 02 | `02_preprocess.ipynb` | ⚠️ Early approach — MediaPipe landmark extraction (ANN) |
| 03 | `03_augment.ipynb` | ⚠️ Early approach — Landmark augmentation (ANN) |
| 04 | `04_model.ipynb` | ⚠️ Early approach — ANN architecture definition |
| 05 | `05_train_mobilenet.ipynb` | ✅ **Final model** — MobileNetV2 training (99.83% accuracy) |
| 06 | `06_evaluate.ipynb` | ✅ Model evaluation and confusion matrix |
| 07 | `07_camera_test.ipynb` | ✅ Live camera inference test |

> **Note:** Notebooks 02, 03, and 04 document the early landmark-based ANN exploration. The final production model uses MobileNetV2 (notebook 05).

### Production Files (`Agents/` + `API/`)

The agent pipeline and FastAPI server power the live application. They import the trained model and run inference in real time.

> **In short:** Use notebooks to train. Use agents + API to run the app.

---

## Dataset

Signify uses the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

### Citation

If you use this project or dataset, please cite:

@misc{asl_alphabet_kaggle,
  author    = {Akash},
  title     = {ASL Alphabet},
  year      = {2018},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/grassknoted/asl-alphabet}
}

### Download Instructions

1. Place your Kaggle API key at `~/.kaggle/kaggle.json`.
   Get it from: [https://www.kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings)

2. Download and extract the dataset:
   ```bash
   kaggle datasets download -d grassknoted/asl-alphabet
   unzip asl-alphabet.zip -d data/raw/kaggle_asl/
   ```

3. Your data folder should look like:
   ```
   data/raw/kaggle_asl/ASL_Alphabet_Dataset/
   └── asl_alphabet_train/   ← 87,000 training images (29 classes)
   ```

> The dataset is NOT committed to this repository — download it from Kaggle using the instructions above.

---

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Python 3.10
- NVIDIA GPU with CUDA 12.1 support (recommended; CPU fallback available)
- Git
- [Ollama](https://ollama.com/download) — local LLM runtime

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
# GPU (recommended):
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU only (fallback):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

**4. Install remaining dependencies**
>>>>>>> 34100ffac8e0d8e4630374a3a240c1a2b2cb7b15
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
Or restore the full conda environment:

```bash
conda env create -f environment.yml   # if provided
conda activate signify
=======
**5. Install Ollama and pull phi3:mini**
```bash
# Download and install Ollama from https://ollama.com/download
# Then pull the model:
ollama pull phi3:mini
```

**6. Create project folders**
```bash
python create_necessary_folders.py
```

**7. Verify your environment**
```bash
python verify_env.py
```

Expected output:
```
✅ Python       3.10.x
✅ PyTorch      2.5.1+cu121  (or CPU version)
✅ CUDA         Available — <your GPU name>  (or Not available — CPU mode)
✅ OpenCV       4.9.0.80
✅ MediaPipe    0.10.14
✅ Whisper      ready
✅ Ollama       ready
✅ All checks passed. Environment is ready.
>>>>>>> 34100ffac8e0d8e4630374a3a240c1a2b2cb7b15
```

---

<<<<<<< HEAD
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
=======
## Running the App

**1. Make sure Ollama is running:**
```bash
ollama serve
```

**2. Start the FastAPI server:**
```bash
conda activate signify
uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Open the web app:**
```
http://localhost:8000/app
```

**4. Interactive API docs:**
```
http://localhost:8000/docs
http://localhost:8000/redoc
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Server status and pipeline info |
| GET | `/health` | Health check |
| GET | `/app` | Serve the web interface |
| POST | `/frame` | Submit a camera frame for inference |
| POST | `/generate` | Convert accumulated letters to a sentence via phi3:mini |
| POST | `/speak` | Speak the generated sentence aloud |
| POST | `/reset` | Clear all buffers |
| GET | `/state` | Get current accumulated text state |

---

*Built with PyTorch, MediaPipe, FastAPI, and a lot of hand signs.*
>>>>>>> 34100ffac8e0d8e4630374a3a240c1a2b2cb7b15
