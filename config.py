import os
import torch

# ── Base directory — works on any machine ─────────────────────────────────────
# Automatically detects the project root regardless of where it is cloned
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset paths ─────────────────────────────────────────────────────────────
KAGGLE_ASL_PATH = os.path.join(BASE_DIR, "data", "raw", "kaggle_asl", "ASL_Alphabet_Dataset", "asl_alphabet_train")
ASL_CITIZEN_PATH = os.path.join(BASE_DIR, "data", "raw", "asl_citizen")
WLASL_PATH = os.path.join(BASE_DIR, "data", "raw", "wlasl")

# ── Processed data paths ──────────────────────────────────────────────────────
LANDMARKS_PATH = os.path.join(BASE_DIR, "data", "processed", "landmarks")
LANDMARKS_AUGMENTED_PATH = os.path.join(BASE_DIR, "data", "processed", "landmarks_augmented")

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_PATH = os.path.join(BASE_DIR, "models", "saved")
BEST_MODEL = os.path.join(MODELS_PATH, "best_model_mobilenet.pth")

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CONFIDENCE_THRESHOLD = 0.7
IMAGE_SIZE = 128

# ── ASL Classes ───────────────────────────────────────────────────────────────
ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]
NUM_CLASSES = len(ASL_CLASSES)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Print config on import ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Signify Configuration")
    print(f"Base directory: {BASE_DIR}")
    print(f"Device:         {DEVICE}")
    print(f"Classes:        {NUM_CLASSES}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Image size:     {IMAGE_SIZE}")