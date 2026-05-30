"""
========================================================
How2Sign Label File Generator and Verifier
========================================================

Usage (command line):
  python label_generator.py generate     # regenerate labels for all splits
  python label_generator.py verify       # verify existing labels only
  python label_generator.py all          # generate then verify (default)
  python label_generator.py all --splits train val  # specific splits only

Paths are identical to how2sign_importer.py — no extra configuration needed.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Path config (mirrors how2sign_importer.py)
# ============================================================

BASE_DIR = Path(__file__).parent
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

SPLITS = ["train", "val", "test"]
TARGET_DIM = 225
CSV_SEPARATOR = "\t"


# ============================================================
# Shared helpers
# ============================================================


def _load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        sep=CSV_SEPARATOR,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )


def _col(df: pd.DataFrame, primary: str, fallback: str) -> str:
    return primary if primary in df.columns else fallback


def _build_text_index(df: pd.DataFrame) -> dict[str, str]:
    """Return {SENTENCE_NAME: sentence_text} from a metadata DataFrame."""
    id_col = _col(df, "SENTENCE_NAME", "SENTENCE_ID")
    text_col = _col(df, "SENTENCE", "TRANSLATION")
    index: dict[str, str] = {}
    for row in df.itertuples(index=False):
        name = str(getattr(row, id_col, "")).strip()
        text = str(getattr(row, text_col, "")).strip()
        if name and text and text.lower() != "nan":
            index[name] = text
    return index


# ============================================================
# 1. Generate label files
# ============================================================


def generate_labels(splits: list[str] = SPLITS) -> dict[str, dict]:
    """
    Scan output/keypoints/{split}/*.npy, match CSV sentences, and write
    output/labels/{split}.txt.

    The .npy stem == SENTENCE_NAME (as produced by how2sign_importer.py).
    Lines are written in strict alphabetical order of the stem so they align
    with sorted(glob("keypoints/{split}/*.npy")) in the training script.
    Rows with no CSV match get a [MISSING:name] placeholder to keep the
    line count equal to the .npy count.
    """
    labels_dir = OUTPUT_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"[generate] split: {split}")

        kp_dir = OUTPUT_DIR / "keypoints" / split
        if not kp_dir.exists():
            print(f"  SKIP: keypoints dir not found: {kp_dir}")
            report[split] = {"status": "skipped", "reason": "no keypoints dir"}
            continue

        npy_files = sorted(kp_dir.glob("*.npy"))
        if not npy_files:
            print(f"  SKIP: no .npy files found")
            report[split] = {"status": "skipped", "reason": "no npy files"}
            continue
        print(f"  .npy files found: {len(npy_files)}")

        csv_path = CSV_PATHS[split]
        if not csv_path.exists():
            print(f"  SKIP: CSV not found: {csv_path}")
            report[split] = {"status": "skipped", "reason": "csv missing"}
            continue

        df = _load_csv(csv_path)
        txt_idx = _build_text_index(df)
        print(f"  CSV loaded: {len(df)} rows -> {len(txt_idx)} valid entries")

        matched = 0
        unmatched = 0
        lines: list[str] = []
        correspondence: list[dict] = []  # [{sentence_name, sentence}]

        for npy_path in npy_files:
            name = npy_path.stem
            text = txt_idx.get(name)
            if text is None:
                # strip view suffix and retry (holistic_npy naming variant)
                short = name.split("-rgb_front")[0].split("-rgb_side")[0]
                text = txt_idx.get(short)
            if text:
                lines.append(text)
                correspondence.append({"sentence_name": name, "sentence": text})
                matched += 1
            else:
                lines.append(f"[MISSING:{name}]")
                unmatched += 1
                print(f"  WARNING: '{name}' not in CSV, writing placeholder")

        label_path = labels_dir / f"{split}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        report[split] = {
            "status": "ok",
            "npy_count": len(npy_files),
            "matched": matched,
            "unmatched": unmatched,
            "label_path": str(label_path),
            "correspondence": correspondence,
        }
        print(f"  OK: wrote {len(lines)} lines -> {label_path}")
        print(f"      matched={matched}  placeholders={unmatched}")

    _print_generate_summary(report)
    _save_generate_report(report)
    return report


def _print_generate_summary(report: dict):
    print(f"\n{'=' * 50}")
    print("Generate summary:")
    for split, info in report.items():
        if info["status"] == "ok":
            print(
                f"  {split:5s}: npy={info['npy_count']:6d}  "
                f"matched={info['matched']:6d}  placeholders={info['unmatched']:4d}"
            )
        else:
            print(f"  {split:5s}: skipped ({info['reason']})")


def _save_generate_report(report: dict):
    data = {
        split: {
            "status": info["status"],
            "npy_count": info.get("npy_count"),
            "matched": info.get("matched"),
            "unmatched": info.get("unmatched"),
            "label_path": info.get("label_path"),
            "correspondence": info.get("correspondence", []),
        }
        for split, info in report.items()
    }
    path = OUTPUT_DIR / "label_generate_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generate report saved: {path}")


# ============================================================
# 2. Verify label files
# ============================================================


class VerifyResult:
    def __init__(self, split: str):
        self.split = split
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.npy_count: int = 0
        self.label_count: int = 0
        self.missing_n: int = 0
        self.missing_entries: list[dict] = []
        self.dim_errors: list[str] = []
        self.correspondence_checked: int = 0
        self.correspondence_mismatches: list[dict] = []
        self.first_stem: str = ""
        self.first_label: str = ""
        self.last_stem: str = ""
        self.last_label: str = ""
        self.sample_shape: tuple = ()

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def verify_labels(splits: list[str] = SPLITS) -> bool:
    """
    Run comprehensive checks on output/labels/{split}.txt.

    Checks:
      1.  Label file exists
      2.  Keypoints directory exists and contains .npy files
      3.  Label line count == .npy file count
      4.  No empty label lines
      5.  No placeholder lines ([MISSING:...])
      6.  .npy dimension spot-check (10 random samples, shape[1] must == 225)
      7.  SENTENCE_NAME <-> SENTENCE correspondence against CSV
          (each label line must match the CSV sentence for that .npy stem)
    """
    import random

    print(f"\n{'=' * 50}")
    print("Verifying label files...")

    results: list[VerifyResult] = []

    for split in splits:
        res = VerifyResult(split)
        kp_dir = OUTPUT_DIR / "keypoints" / split
        label_path = OUTPUT_DIR / "labels" / f"{split}.txt"

        # ── 1. Existence ────────────────────────────────────────
        if not label_path.exists():
            res.error(f"label file not found: {label_path}")
            results.append(res)
            _print_split_result(res)
            continue

        if not kp_dir.exists():
            res.error(f"keypoints dir not found: {kp_dir}")
            results.append(res)
            _print_split_result(res)
            continue

        npy_files = sorted(kp_dir.glob("*.npy"))
        if not npy_files:
            res.error(f"no .npy files in keypoints dir: {kp_dir}")
            results.append(res)
            _print_split_result(res)
            continue

        # ── 2. Read labels ──────────────────────────────────────
        with open(label_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        labels = [ln.rstrip("\n") for ln in raw_lines]
        if labels and labels[-1] == "":
            labels = labels[:-1]

        # ── 3. Count alignment ──────────────────────────────────
        n_npy = len(npy_files)
        n_labels = len(labels)

        if n_npy != n_labels:
            res.error(f"npy count ({n_npy}) != label line count ({n_labels})")
            results.append(res)
            _print_split_result(res)
            continue

        # ── 4. Empty lines ──────────────────────────────────────
        empty_idxs = [i for i, ln in enumerate(labels) if not ln.strip()]
        if empty_idxs:
            res.error(
                f"{len(empty_idxs)} empty label line(s); "
                f"first indices (0-based): {empty_idxs[:5]}"
            )

        # ── 5. Placeholder lines ────────────────────────────────
        missing_entries = [
            {"index": i, "placeholder": ln}
            for i, ln in enumerate(labels)
            if ln.startswith("[MISSING:")
        ]
        if missing_entries:
            res.warn(
                f"{len(missing_entries)} placeholder line(s) ([MISSING:...]); "
                f"first: {missing_entries[0]['placeholder']}"
            )

        # ── 6. .npy dimension spot-check ────────────────────────
        sample_npy = random.sample(npy_files, min(10, n_npy))
        dim_errors: list[str] = []

        for p in sample_npy:
            try:
                arr = np.load(str(p))
            except Exception as e:
                dim_errors.append(f"{p.name}: load failed ({e})")
                continue
            if arr.ndim != 2:
                dim_errors.append(f"{p.name}: wrong ndim={arr.ndim}")
            elif arr.shape[1] != TARGET_DIM:
                dim_errors.append(
                    f"{p.name}: expected dim={TARGET_DIM}, got dim={arr.shape[1]}"
                )

        for e in dim_errors:
            res.error(f"npy dimension: {e}")

        # ── 8. SENTENCE_NAME <-> SENTENCE correspondence ────────
        csv_path = CSV_PATHS[split]
        correspondence_mismatches: list[dict] = []
        correspondence_checked = 0

        if csv_path.exists():
            df = _load_csv(csv_path)
            txt_idx = _build_text_index(df)

            for idx, (npy_path, label_line) in enumerate(zip(npy_files, labels)):
                name = npy_path.stem
                expected = txt_idx.get(name)
                if expected is None:
                    short = name.split("-rgb_front")[0].split("-rgb_side")[0]
                    expected = txt_idx.get(short)

                if expected is None:
                    # CSV has no entry — already flagged as placeholder
                    continue

                correspondence_checked += 1
                if label_line != expected:
                    correspondence_mismatches.append(
                        {
                            "index": idx,
                            "sentence_name": name,
                            "label_in_file": label_line[:100],
                            "expected_from_csv": expected[:100],
                        }
                    )

            if correspondence_mismatches:
                res.error(
                    f"{len(correspondence_mismatches)} SENTENCE_NAME<->SENTENCE mismatch(es); "
                    f"first: index={correspondence_mismatches[0]['index']} "
                    f"name='{correspondence_mismatches[0]['sentence_name']}'"
                )
        else:
            res.warn(f"CSV not found, skipping correspondence check: {csv_path}")
            correspondence_checked = -1  # sentinel: unavailable

        # ── Collect metadata for report ─────────────────────────
        first_arr = np.load(str(npy_files[0]))

        res.npy_count = n_npy
        res.label_count = n_labels
        res.missing_n = len(missing_entries)
        res.missing_entries = missing_entries
        res.dim_errors = dim_errors
        res.correspondence_checked = correspondence_checked
        res.correspondence_mismatches = correspondence_mismatches
        res.first_stem = npy_files[0].stem
        res.first_label = labels[0][:70]
        res.last_stem = npy_files[-1].stem
        res.last_label = labels[-1][:70]
        res.sample_shape = first_arr.shape

        results.append(res)
        _print_split_result(res)

    all_ok = _print_global_summary(results)
    _save_verify_report(results, all_ok)
    return all_ok


def _print_split_result(res: VerifyResult):
    icon = "OK" if res.ok else "FAIL"
    print(f"\n  [{res.split}] {icon}")
    if hasattr(res, "npy_count"):
        print(
            f"    npy count    : {res.npy_count}  |  sample shape: {res.sample_shape}"
        )
        print(f"    label lines  : {res.label_count}")
        checked = getattr(res, "correspondence_checked", 0)
        mismatches = len(getattr(res, "correspondence_mismatches", []))
        if checked >= 0:
            print(f"    CSV checked  : {checked}  |  mismatches: {mismatches}")
        if res.missing_n:
            print(f"    placeholders : {res.missing_n}")
        print(f"    first npy    : {res.first_stem}")
        print(f'    first label  : "{res.first_label}"')
        print(f"    last npy     : {res.last_stem}")
        print(f'    last label   : "{res.last_label}"')
    for e in res.errors:
        print(f"    ERROR   : {e}")
    for w in res.warnings:
        print(f"    WARNING : {w}")


def _print_global_summary(results: list[VerifyResult]) -> bool:
    all_ok = all(r.ok for r in results)
    print(f"\n{'=' * 50}")
    print("Verification summary:")
    for r in results:
        total = getattr(r, "npy_count", 0)
        missing = getattr(r, "missing_n", 0)
        mismatches = len(getattr(r, "correspondence_mismatches", []))
        status = "OK  " if r.ok else "FAIL"
        print(
            f"  {r.split:5s} [{status}]  "
            f"total={total:6d}  placeholders={missing:4d}  "
            f"csv_mismatches={mismatches:4d}  "
            f"errors={len(r.errors)}  warnings={len(r.warnings)}"
        )
    verdict = "All checks passed." if all_ok else "Errors found — see report."
    print(f"\n{verdict}")
    return all_ok


def _save_verify_report(results: list[VerifyResult], all_ok: bool):
    data = {
        "all_ok": all_ok,
        "splits": {
            r.split: {
                "ok": r.ok,
                "errors": r.errors,
                "warnings": r.warnings,
                "npy_count": getattr(r, "npy_count", None),
                "label_count": getattr(r, "label_count", None),
                "placeholder_count": getattr(r, "missing_n", None),
                "placeholder_entries": getattr(r, "missing_entries", []),
                "dim_errors": getattr(r, "dim_errors", []),
                "correspondence": {
                    "checked": getattr(r, "correspondence_checked", None),
                    "mismatches": len(getattr(r, "correspondence_mismatches", [])),
                    "mismatch_entries": getattr(r, "correspondence_mismatches", []),
                },
                "first_stem": getattr(r, "first_stem", None),
                "first_label": getattr(r, "first_label", None),
                "last_stem": getattr(r, "last_stem", None),
                "last_label": getattr(r, "last_label", None),
            }
            for r in results
        },
    }
    path = OUTPUT_DIR / "label_verify_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Verify report saved: {path}")


# ============================================================
# Entry point
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="How2Sign label file generator and verifier"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["generate", "verify", "all"],
        help="generate=write labels only  verify=check only  all=both (default)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        metavar="SPLIT",
        help="splits to process (default: train val test)",
    )
    args = parser.parse_args()

    if args.action in ("generate", "all"):
        generate_labels(args.splits)

    if args.action in ("verify", "all"):
        ok = verify_labels(args.splits)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
