import os
import sys
import zipfile
import urllib.request
from pathlib import Path

url = "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
zip_path = data_dir / "CheXpert-v1.0-small.zip"
extract_dir = data_dir / "CheXpert-v1.0-small"


def download(url, out_path):
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0

        sys.stdout.write(
            f"\rDownloading: {percent:6.2f}% ({mb_done:8.1f} MB / {mb_total:8.1f} MB)"
        )
        sys.stdout.flush()

    print(f"Downloading {out_path} ...")
    urllib.request.urlretrieve(url, out_path, reporthook=progress)
    print("\nDownload complete")


def unzip(zip_path, extract_to):
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to.parent)
    print("Extraction complete")


def main():
    data_dir.mkdir(exist_ok=True)

    if not zip_path.exists():
        download(url, zip_path)

    unzip(zip_path, extract_dir)

    print("\nDataset located at:")
    print(extract_dir)


if __name__ == "__main__":
    main()
