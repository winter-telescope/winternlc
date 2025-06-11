"""
Module to download the correction files from Zenodo.
"""

import logging
import subprocess
from pathlib import Path

from winternlc.config import get_correction_dir
from winternlc.zenodo import ZENODO_URL_MAP, get_zenodo_url

logger = logging.getLogger(__name__)


def check_files(version: str) -> bool:
    """
    Check if the correction files are present in the correction directory.
    """
    corrections_dir = get_correction_dir(version)
    files = list(corrections_dir.glob("*.npy"))
    return len(files) > 0


def rename_files(path: Path):
    """
    Normalize filenames by removing '_extXYZ' and duplicate '.npy' extensions,
    but skip files that are already in final form (e.g., bad_pixel_mask_board_1.npy).
    """
    old_name = path.name
    base = path.stem  # removes last .npy

    # Handle case where base still ends with '.npy' (double suffix)
    if base.endswith(".npy"):
        base = Path(base).stem

    # Skip if already in clean form (no '_ext' and single '.npy')
    if "_ext" not in base:
        logger.debug(f"Skipping rename: {old_name} is already clean")
        return

    new_base = base.split("_ext")[0]
    new_name = new_base + ".npy"

    new_path = path.with_name(new_name)
    if new_path.exists():
        logger.warning(f"Skipping rename: {new_path.name} already exists")
        return

    logger.info(f"Renaming {old_name} → {new_name}")
    path.rename(new_path)


def download_files(version: str):
    """
    Download the correction files from Zenodo.

    :return: None
    """

    zenodo_url = get_zenodo_url(version)

    record = Path(zenodo_url).parent.name

    corrections_dir = get_correction_dir(version)

    out_file = corrections_dir / f"{record}.zip"

    if not out_file.exists():
        command = f"wget {zenodo_url} -O {out_file}"
        logging.info(f"Downloading correction files from {zenodo_url}")
        subprocess.run(command, shell=True, check=True)
    else:
        logging.debug(f"Correction files already downloaded to {out_file}")

    assert out_file.exists(), f"Download failed: {out_file}"

    command = f"unzip {out_file} -d {corrections_dir}"
    logging.info(f"Unzipping correction files to {corrections_dir}")
    subprocess.run(command, shell=True, check=True)

    assert check_files(version), f"Unzipping failed: {corrections_dir}"

    for path in corrections_dir.glob("*.npy"):
        rename_files(path)


def check_for_files():
    """
    Check if the correction files are present in the correction directory.

    :return: None
    """

    for version in ZENODO_URL_MAP.keys():
        if not check_files(version):
            logger.info(f"Correction files found for {version}")
            download_files(version)
        else:
            logger.debug(f"Correction files already present in {version}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    check_for_files()
