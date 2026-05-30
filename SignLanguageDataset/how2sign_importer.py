"""
========================================================
How2Sign Dataset Importer
→ Output format compatible with sign_language_keypoints_to_text.py
========================================================

Extracts MediaPipe Holistic keypoints from verified_videos/*.mp4 in real time.
Label file generation and verification are delegated to label_generator.py.

Expected directory layout:

  <script dir>/
  ├── How2Sign/
  │   └── sentence_level/
  │       ├── train/how2sign_verified_train.csv
  │       ├── val/how2sign_verified_val.csv
  │       └── test/how2sign_verified_test.csv
  │
  ├── verified_videos/*.mp4
  │
  └── how2sign_importer.py

Output layout (compatible with sign_language_keypoints_to_text.py):
  output/
  ├── keypoints/
  │   ├── train/   ← .npy files, shape (T, 225), one per sentence
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train.txt  ← one English sentence per line
      ├── val.txt    ← order matches sorted(glob("keypoints/{split}/*.npy"))
      └── test.txt

Dependencies: pip install pandas numpy tqdm av mediapipe>=0.10
              holistic_landmarker.task is downloaded automatically on first run
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from label_generator import generate_labels, verify_labels

# ============================================================
# Configuration
# ============================================================

# Maximum samples per split (None = no limit; set to e.g. 100 for debugging)
MAX_SAMPLES = None

BASE_DIR = Path(__file__).parent
VERIFIED_VIDEOS = BASE_DIR / "verified_videos"
OUTPUT_DIR = BASE_DIR / "output"

CSV_PATHS = {
    "train": BASE_DIR
    / "How2Sign"
    / "sentence_level"
    / "train"
    / "how2sign_verified_train.csv",
    "val": BASE_DIR
    / "How2Sign"
    / "sentence_level"
    / "val"
    / "how2sign_verified_val.csv",
    "test": BASE_DIR
    / "How2Sign"
    / "sentence_level"
    / "test"
    / "how2sign_verified_test.csv",
}

# MediaPipe model file (downloaded automatically on first run)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
MODEL_PATH = BASE_DIR / "holistic_landmarker.task"


# ============================================================
# Keypoint layout (225-dim)
# ============================================================
#   [0  : 99 ] pose       (33 landmarks × 3 coords)
#   [99 : 162] left_hand  (21 landmarks × 3 coords)
#   [162: 225] right_hand (21 landmarks × 3 coords)

TARGET_DIM = 225
SPLITS = ["train", "val", "test"]
CSV_SEPARATOR = "\t"


# ============================================================
# Shared utilities
# ============================================================


def make_dirs(output_dir: Path):
    for split in SPLITS:
        (output_dir / "keypoints" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Output directories ready: {output_dir}")


def load_metadata_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=CSV_SEPARATOR,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    print(f"  CSV loaded: {csv_path.name} -> {len(df)} rows")
    return df


def validate_npy(arr: np.ndarray, name: str) -> bool:
    if arr.ndim != 2:
        print(f"  Skipping {name}: wrong shape {arr.shape}")
        return False
    T, _ = arr.shape
    if T < 5:
        print(f"  Skipping {name}: too few frames ({T})")
        return False
    if np.isnan(arr).all():
        print(f"  Skipping {name}: all NaN")
        return False
    return True


def save_manifest(stats: dict):
    manifest = {
        "source": "verified_videos_mp4_extracted",
        "target_dim": TARGET_DIM,
        "splits": stats,
        "format": {
            "keypoints": "numpy (.npy), shape=(T, 225), float32",
            "labels": "plain text, UTF-8; "
            "line i corresponds to sorted(glob('keypoints/{split}/*.npy'))[i]",
        },
        "dim_layout": {
            "0:99": "pose       (33 landmarks × 3 coords)",
            "99:162": "left_hand  (21 landmarks × 3 coords)",
            "162:225": "right_hand (21 landmarks × 3 coords)",
        },
        "paths": {
            "output_dir": str(OUTPUT_DIR),
            **{f"csv_{s}": str(CSV_PATHS[s]) for s in SPLITS},
        },
    }
    path = OUTPUT_DIR / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved: {path}")


# ============================================================
# MediaPipe extraction helpers
# ============================================================


def _ensure_model_file() -> Path:
    import urllib.request

    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading Holistic model -> {MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("Download complete.")
    return MODEL_PATH


def _build_landmarker():
    """
    Build a MediaPipe Tasks HolisticLandmarker in VIDEO mode.
    VIDEO mode requires strictly increasing timestamp_ms, so each video
    must use its own landmarker instance.
    """
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )
    from mediapipe.tasks.python.vision.holistic_landmarker import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
    )

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_ensure_model_file())),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(options)


def _landmarks_to_flat(landmarks, expected_count: int) -> list:
    """
    Flatten a list[NormalizedLandmark] from the MediaPipe Tasks API into
    [x0, y0, z0, x1, y1, z1, ...] of length expected_count × 3.
    Returns all-zeros when landmarks are missing or have unexpected count.
    """
    if landmarks and len(landmarks) == expected_count:
        flat = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        return flat
    return [0.0] * (expected_count * 3)


def _extract_keypoints_from_mp4(mp4_path: Path) -> np.ndarray:
    """
    Extract per-frame keypoints from a local .mp4, returning shape (T, 225) float32.

    A fresh landmarker instance is created per video to satisfy VIDEO mode's
    requirement that timestamps start from 0 and increase monotonically.

    Keypoint layout (225-dim):
        [0  : 99 ] pose       (33 × 3)
        [99 : 162] left_hand  (21 × 3)
        [162: 225] right_hand (21 × 3)
    """
    import av
    import mediapipe as mp

    landmarker = _build_landmarker()
    frames_kp = []

    try:
        container = av.open(str(mp4_path))
        video_stream = next(iter(container.streams.video), None)
        if video_stream is None:
            return np.zeros((0, TARGET_DIM), dtype=np.float32)

        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0

        for frame_idx, av_frame in enumerate(container.decode(video=0)):
            img_rgb = av_frame.to_ndarray(format="rgb24")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            timestamp = int(frame_idx * 1000 / fps)  # ms, strictly increasing

            result = landmarker.detect_for_video(mp_image, timestamp)

            # Concatenate 225-dim: pose(99) + left_hand(63) + right_hand(63)
            kp = (
                _landmarks_to_flat(result.pose_landmarks, 33)
                + _landmarks_to_flat(result.left_hand_landmarks, 21)
                + _landmarks_to_flat(result.right_hand_landmarks, 21)
            )
            frames_kp.append(kp)

        container.close()
    finally:
        landmarker.close()

    if not frames_kp:
        return np.zeros((0, TARGET_DIM), dtype=np.float32)
    return np.array(frames_kp, dtype=np.float32)


# ============================================================
# Main import function
# ============================================================


def import_from_mp4():
    """
    Extract MediaPipe Holistic keypoints from verified_videos/*.mp4.

    Matching rule: mp4 stem == SENTENCE_NAME
    i.e. verified_videos/<SENTENCE_NAME>.mp4 maps to the same-named CSV row.

    After keypoint extraction, label files are generated and verified via
    label_generator.generate_labels() and label_generator.verify_labels().
    """
    try:
        import av  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install av mediapipe>=0.10")
        return

    if not VERIFIED_VIDEOS.exists():
        print(f"ERROR: verified_videos directory not found: {VERIFIED_VIDEOS}")
        return

    make_dirs(OUTPUT_DIR)

    # Build a global SENTENCE_NAME -> mp4 path index (shared across splits)
    mp4_index: dict[str, Path] = {p.stem: p for p in VERIFIED_VIDEOS.glob("*.mp4")}
    print(f"verified_videos: {len(mp4_index)} .mp4 file(s) found")

    stats = {}

    for split in SPLITS:
        print(f"\n{'=' * 50}")
        print(f"Processing split: {split}")

        # 1. Load CSV metadata
        csv_path = CSV_PATHS[split]
        if not csv_path.exists():
            print(f"  SKIP: CSV not found: {csv_path}")
            continue
        meta_df = load_metadata_csv(csv_path)

        id_col = (
            "SENTENCE_NAME" if "SENTENCE_NAME" in meta_df.columns else "SENTENCE_ID"
        )
        text_col = "SENTENCE" if "SENTENCE" in meta_df.columns else "TRANSLATION"

        # 2. Extract keypoints row by row
        out_kp_dir = OUTPUT_DIR / "keypoints" / split
        collected = []
        skip_count = 0

        rows = meta_df.itertuples(index=False)
        if MAX_SAMPLES:
            import itertools

            rows = itertools.islice(rows, MAX_SAMPLES)

        for row in tqdm(list(rows), desc=f"  {split}"):
            sentence_name = str(getattr(row, id_col, "")).strip()
            sentence_text = str(getattr(row, text_col, "")).strip()

            if not sentence_text or sentence_text in ("nan", ""):
                skip_count += 1
                continue

            npy_out_path = out_kp_dir / f"{sentence_name}.npy"

            # Already extracted — record and skip re-decoding
            if npy_out_path.exists():
                collected.append(sentence_name)
                continue

            mp4_path = mp4_index.get(sentence_name)
            if mp4_path is None:
                skip_count += 1
                continue

            try:
                kp = _extract_keypoints_from_mp4(mp4_path)
            except Exception as e:
                print(f"  Extraction failed {mp4_path.name}: {e}")
                skip_count += 1
                continue

            if not validate_npy(kp, sentence_name):
                skip_count += 1
                continue

            np.save(str(npy_out_path), kp)
            collected.append(sentence_name)

        ok_count = len(collected)
        stats[split] = {"ok": ok_count, "skipped": skip_count}
        print(f"  OK: {ok_count} extracted  |  skipped: {skip_count}")
        print(f"  Keypoints -> {out_kp_dir}")

        # 3. Generate label file from the extracted keypoints on disk
        generate_labels([split])

    _print_summary(stats)
    save_manifest(stats)


# ============================================================
# Helpers
# ============================================================


def _print_summary(stats: dict):
    print(f"\n{'=' * 50}")
    print("Import complete:")
    for split, s in stats.items():
        print(f"  {split:5s}: {s['ok']:6d} ok  {s['skipped']:5d} skipped")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import_from_mp4()
    verify_labels()
