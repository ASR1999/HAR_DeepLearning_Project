import requests
import zipfile
import io
import os
import shutil

# URLs
UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAPT_URL = "https://archive.ics.uci.edu/static/public/231/human+activities+and+postural+transitions.zip"

# Paths
DATA_DIR = "./data"
STD_UCI_DIR = os.path.join(DATA_DIR, "uci_har")
STD_HAPT_DIR = os.path.join(DATA_DIR, "hapt")


def _download_zip(url: str) -> zipfile.ZipFile:
    print(f"Downloading: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(resp.content))


def _safe_extract(zf: zipfile.ZipFile, dest_root: str):
    os.makedirs(dest_root, exist_ok=True)
    zf.extractall(dest_root)


def _find_subdir_with_files(root: str, required_rel_paths: list[str]) -> str | None:
    for current_root, dirs, files in os.walk(root):
        # Create a set of relative paths from current_root
        rel_candidates = []
        for rel in required_rel_paths:
            candidate = os.path.join(current_root, rel)
            rel_candidates.append(candidate)
        if all(os.path.exists(p) for p in rel_candidates):
            return current_root
    return None


def _normalize_uci(extract_root: str):
    # UCI signals and features live under a folder containing train/ and test/
    required = [
        os.path.join("train", "y_train.txt"),
        os.path.join("test", "y_test.txt"),
    ]
    src_root = _find_subdir_with_files(extract_root, required)
    if src_root is None:
        raise RuntimeError("Could not locate UCI HAR train/test after extraction")
    if os.path.isdir(STD_UCI_DIR):
        shutil.rmtree(STD_UCI_DIR)
    shutil.copytree(src_root, STD_UCI_DIR)
    print(f"Standardized UCI HAR to: {STD_UCI_DIR}")


def _normalize_hapt(extract_root: str):
    # HAPT has Train/Test with X_train.txt etc.
    required = [
        os.path.join("Train", "X_train.txt"),
        os.path.join("Test", "X_test.txt"),
    ]
    src_root = _find_subdir_with_files(extract_root, required)
    if src_root is None:
        raise RuntimeError("Could not locate HAPT Train/Test after extraction")
    if os.path.isdir(STD_HAPT_DIR):
        shutil.rmtree(STD_HAPT_DIR)
    shutil.copytree(src_root, STD_HAPT_DIR)
    print(f"Standardized HAPT to: {STD_HAPT_DIR}")


def download_and_extract_all():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        # UCI HAR
        with _download_zip(UCI_URL) as zf:
            tmp_uci = os.path.join(DATA_DIR, "_tmp_uci")
            if os.path.isdir(tmp_uci):
                shutil.rmtree(tmp_uci)
            _safe_extract(zf, tmp_uci)
            _normalize_uci(tmp_uci)
            shutil.rmtree(tmp_uci, ignore_errors=True)

        # HAPT
        with _download_zip(HAPT_URL) as zf:
            tmp_hapt = os.path.join(DATA_DIR, "_tmp_hapt")
            if os.path.isdir(tmp_hapt):
                shutil.rmtree(tmp_hapt)
            _safe_extract(zf, tmp_hapt)
            _normalize_hapt(tmp_hapt)
            shutil.rmtree(tmp_hapt, ignore_errors=True)

        print("\nAll datasets downloaded and standardized:")
        print(f" - UCI HAR -> {STD_UCI_DIR}")
        print(f" - HAPT    -> {STD_HAPT_DIR}")

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except zipfile.BadZipFile:
        print("Error: A downloaded file is not a valid ZIP")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    download_and_extract_all()