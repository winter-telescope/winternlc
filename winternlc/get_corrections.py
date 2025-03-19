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
    Remove useless part of the filename
    """
    old_name = path.name
    new_name = str(old_name).split("_ext")[0] + path.suffix
    logger.info(f"Renaming {old_name} to {new_name}")
    path.rename(path.with_name(new_name))


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
