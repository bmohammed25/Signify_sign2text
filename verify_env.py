"""
verify_env.py
Signify — ASL Sign Language Communication App
Fanshawe College Capstone 2026 — Group 7

Run this script after setting up your environment to confirm
all required libraries are installed and CUDA is available.

Usage:
    python verify_env.py
"""

import sys

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"

passed = []
failed = []

def check(label, fn):
    try:
        result = fn()
        print(f"  {GREEN}✅ {label:<20}{RESET} {result}")
        passed.append(label)
    except Exception as e:
        print(f"  {RED}❌ {label:<20}{RESET} {e}")
        failed.append(label)

# ── Python ────────────────────────────────────────────────────
print(f"\n{BOLD}── Python ──────────────────────────────────────────{RESET}")
check("Python",
    lambda: f"{sys.version.split()[0]}  {'✔ 3.10' if sys.version_info[:2] == (3,10) else '⚠ expected 3.10'}")

# ── Deep Learning ─────────────────────────────────────────────
print(f"\n{BOLD}── Deep Learning ───────────────────────────────────{RESET}")
def check_torch():
    import torch
    version = torch.__version__
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "not available"
    return f"{version}  |  CUDA: {'✔' if cuda else '✘'}  |  GPU: {gpu}"

check("PyTorch", check_torch)

def check_torchvision():
    import torchvision
    return torchvision.__version__

check("torchvision", check_torchvision)

# ── Computer Vision ───────────────────────────────────────────
print(f"\n{BOLD}── Computer Vision ─────────────────────────────────{RESET}")
check("OpenCV",    lambda: __import__("cv2").__version__)
check("MediaPipe", lambda: __import__("mediapipe").__version__)

# ── Data & Math ───────────────────────────────────────────────
print(f"\n{BOLD}── Data & Math ─────────────────────────────────────{RESET}")
check("NumPy",        lambda: __import__("numpy").__version__)
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("Matplotlib",   lambda: __import__("matplotlib").__version__)
check("Seaborn",      lambda: __import__("seaborn").__version__)
check("Pillow",       lambda: __import__("PIL").__version__)
check("tqdm",         lambda: __import__("tqdm").__version__)

# ── Speech ────────────────────────────────────────────────────
print(f"\n{BOLD}── Speech ──────────────────────────────────────────{RESET}")
check("SpeechRecognition", lambda: __import__("speech_recognition").__version__)
check("gTTS",              lambda: __import__("gtts").__version__)
check("pygame",            lambda: __import__("pygame").__version__)
check("pydub",             lambda: __import__("pydub").__version__)

# ── Whisper ───────────────────────────────────────────────────
print(f"\n{BOLD}── Whisper ─────────────────────────────────────────{RESET}")
def check_whisper():
    import whisper
    models = whisper.available_models()
    return f"ready  |  available models: {', '.join(list(models)[:4])}..."
check("Whisper", check_whisper)

# ── Ollama ────────────────────────────────────────────────────
print(f"\n{BOLD}── Ollama ──────────────────────────────────────────{RESET}")
def check_ollama():
    import ollama
    models = ollama.list()
    names = [m.model for m in models.models] if models.models else []
    return f"ready  |  pulled models: {', '.join(names) if names else 'none pulled yet'}"
check("Ollama", check_ollama)

# ── NLP ───────────────────────────────────────────────────────
print(f"\n{BOLD}── NLP ─────────────────────────────────────────────{RESET}")
check("transformers",  lambda: __import__("transformers").__version__)
check("sentencepiece", lambda: __import__("sentencepiece").__version__)

# ── Dataset & Kaggle ──────────────────────────────────────────
print(f"\n{BOLD}── Dataset & Kaggle ────────────────────────────────{RESET}")
check("kaggle",    lambda: __import__("kaggle").__version__)
check("kagglehub", lambda: __import__("kagglehub").__version__)

# ── Utilities ─────────────────────────────────────────────────
print(f"\n{BOLD}── Utilities ───────────────────────────────────────{RESET}")
check("aiofiles",   lambda: __import__("aiofiles").__version__)
check("python-jose",lambda: __import__("jose").__version__)

# ── Summary ───────────────────────────────────────────────────
total = len(passed) + len(failed)
print(f"\n{BOLD}── Summary ─────────────────────────────────────────{RESET}")
print(f"  {GREEN}{len(passed)}/{total} checks passed{RESET}")

if failed:
    print(f"  {RED}Failed:{RESET}")
    for f in failed:
        print(f"    {RED}✘ {f}{RESET}")
    print(f"\n  {YELLOW}Run: pip install -r requirements.txt{RESET}")
    print(f"  {YELLOW}For PyTorch CUDA: see environment/Local_Environment_Setup.docx{RESET}\n")
else:
    print(f"  {GREEN}{BOLD}✅ All checks passed. Environment is ready for Signify.{RESET}\n")
