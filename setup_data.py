import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd


def ensure_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_dir / "kaggle.json"
    if not kaggle_json_path.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Put your kaggle.json in ~/.kaggle/ and set permissions."
        )
    # Make sure permissions are strict on *nix
    try:
        kaggle_json_path.chmod(0o600)
    except Exception:
        # Not fatal on some OSes, just continue
        pass
    print("✓ Kaggle credentials verified.")


def download_dataset(dataset="shuyangli94/this-american-life-podcast-transcriptsalignments",
                     download_dir=Path("./tal_dataset_raw"),
                     extract_dir=Path("./tal_dataset")):
    """
    Uses the kaggle CLI to download the dataset and extracts it.

    Requires: `kaggle` CLI in PATH and valid ~/.kaggle/kaggle.json
    """
    ensure_kaggle_credentials()

    download_dir = Path(download_dir)
    extract_dir = Path(extract_dir)

    # Clean previous runs (optional)
    if download_dir.exists():
        shutil.rmtree(download_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading dataset from Kaggle...")
    try:
        # This downloads the dataset zip(s) to download_dir
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")

    # Extract any archives found into extract_dir
    zip_files = list(download_dir.glob("*.zip"))
    if not zip_files:
        print("Warning: no zip files found in download directory. If the dataset used direct files, they may already be present.")
    else:
        for z in zip_files:
            try:
                print(f"Extracting {z.name} ...")
                shutil.unpack_archive(str(z), extract_dir)
            except shutil.ReadError:
                print(f"Skipping non-archive file: {z}")

    # Also copy any JSON files (if the kaggle CLI already placed files directly)
    for f in download_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".json", ".txt"}:
            shutil.copy2(f, extract_dir / f.name)

    print("✓ Dataset downloaded and extracted to:", extract_dir)
    return extract_dir


def _normalize_transcript_item(item):
    """
    Normalize a single transcript item to a dict with keys we expect.
    Handles both dicts and other shapes gracefully.
    """
    # Standard keys we search for
    audio_path = item.get("audio_filepath") or item.get("audioFile") or item.get("audio")
    transcript_id = item.get("transcript_id") or item.get("id") or item.get("audio_filepath")

    segments = item.get("segments") or item.get("utterances") or item.get("transcript") or []
    # Ensure segments is a list
    if isinstance(segments, dict):
        # sometimes segments could be keyed dict, convert to list
        segments_list = []
        for v in segments.values():
            if isinstance(v, list):
                segments_list.extend(v)
            else:
                segments_list.append(v)
        segments = segments_list

    return audio_path, transcript_id, segments


def load_tal_transcripts(json_path):
    """
    Parse a TAL transcript-aligned JSON file into a pandas DataFrame of segments.

    Expected segment keys: start, end, speaker, text (but function tolerates missing fields).
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # Read JSON file
    with json_path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            # try line-delimited JSON
            fh.seek(0)
            lines = [line.strip() for line in fh if line.strip()]
            if not lines:
                raise
            data = []
            for ln in lines:
                try:
                    data.append(json.loads(ln))
                except json.JSONDecodeError:
                    # give up on this file
                    raise

    rows = []

    # Data may be a dict (single object) or a list of transcript objects
    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            # skip things we cannot interpret
            continue
        audio_path, transcript_id, segments = _normalize_transcript_item(item)

        # If segments is empty but the top-level item looks like one segment, try to interpret
        if not segments and any(k in item for k in ("start", "end", "speaker", "text")):
            # treat the item itself as a single segment
            seg = {
                "start": item.get("start"),
                "end": item.get("end"),
                "speaker": item.get("speaker"),
                "text": item.get("text"),
            }
            segments = [seg]

        for seg in segments:
            if not isinstance(seg, dict):
                # skip unparseable segments
                continue
            rows.append({
                "transcript_id": transcript_id,
                "audio_filepath": audio_path,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker"),
                "text": seg.get("text"),
            })

    df = pd.DataFrame(rows)
    return df


def load_dataset_from_dir(path):
    """
    Find all transcript-aligned JSON files in path and parse them into a single DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")

    # Look for aligned transcript JSON files (train/test/valid)
    json_files = sorted([p for p in path.glob("*.json") if "transcripts-aligned" in p.name])

    if not json_files:
        # As a fallback, try any JSON that looks like transcripts (contains 'transcript' or 'segments')
        json_files = []
        for p in path.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as fh:
                    chunk = fh.read(1024).lower()
                    if "transcript" in chunk or "segments" in chunk or "audio_filepath" in chunk:
                        json_files.append(p)
            except Exception:
                continue

    if not json_files:
        raise FileNotFoundError(f"No transcript JSON files found in {path!s}")

    frames = []
    for jf in json_files:
        print(f"Loading {jf.name} ...")
        try:
            df = load_tal_transcripts(jf)
            print(f"  -> Loaded {len(df)} segments")
            frames.append(df)
        except Exception as e:
            print(f"  ! Skipped {jf.name}: {e}")

    if not frames:
        raise RuntimeError("No JSON transcript files could be parsed successfully.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset shape: {combined.shape}")
    return combined


def main():
    # Allow optional CLI args via environment variables or simple prompt
    print("=== This American Life (TAL) dataset loader ===")

    try:
        dataset_dir = download_dataset()
    except Exception as e:
        print("Error downloading or extracting dataset:", e)
        sys.exit(1)

    try:
        df = load_dataset_from_dir(dataset_dir)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)

    # Quick quality checks
    print("\nPreview:")
    with pd.option_context("display.max_colwidth", 120):
        print(df.head(10))

    # Save to CSV
    out_csv = Path("tal_transcripts.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved processed transcript segments to: {out_csv}")


if __name__ == "__main__":
    main()