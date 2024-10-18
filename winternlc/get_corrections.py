"""
Module to download the correction files from Zenodo.
"""
import logging
import subprocess
from pathlib import Path
from winternlc.config import corrections_dir
from winternlc.zenodo import ZENODO_URL

logger = logging.getLogger(__name__)


def check_files() -> bool:
    """
    Check if the correction files are present in the correction directory.
    """
    files = list(corrections_dir.glob("*.npy"))
    return len(files) > 0


def rename_files(path: Path) -> Path:
    """
    Remove useless part of the filename
    """
    old_name = path.name
    new_name = str(old_name).split("_ext")[0] + path.suffix
    logger.info(f"Renaming {old_name} to {new_name}")
    path.rename(path.with_name(new_name))


def download_files():
    """
    Download the correction files from Zenodo.

    :return: None
    """

    record = Path(ZENODO_URL).parent.name

    out_file = corrections_dir / f"{record}.zip"

    if not out_file.exists():
        command = f"wget {ZENODO_URL} -O {out_file}"
        logging.info(f"Downloading correction files from {ZENODO_URL}")
        subprocess.run(command, shell=True, check=True)
    else:
        logging.debug(f"Correction files already downloaded to {out_file}")

    assert out_file.exists(), f"Download failed: {out_file}"

    command = f"unzip {out_file} -d {corrections_dir}"
    logging.info(f"Unzipping correction files to {corrections_dir}")
    subprocess.run(command, shell=True, check=True)

    assert check_files(), f"Unzipping failed: {corrections_dir}"

    for path in corrections_dir.glob("*.npy"):
        rename_files(path)


def check_for_files():
    """
    Check if the correction files are present in the correction directory.

    :return: None
    """
    if not check_files():
        logger.info(f"Correction files not found in {corrections_dir}")
        download_files()
    else:
        logger.debug(f"Correction files already present in {corrections_dir}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    check_for_files()
