# versions
available_versions = ["v0.1", "v1.0", "v1.1"]
urls = ["https://zenodo.org/records/13905735", "https://zenodo.org/records/13863497", "https://zenodo.org/records/13905772"]
zenodo_version = "v1.1"  

if zenodo_version in available_versions:
    index = available_versions.index(zenodo_version)
    zenodo_url = urls[index]
else:
    raise ValueError(f"Version {zenodo_version} is not available. Please choose from {available_versions}")

from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

code_dir = Path(__file__).parent

data_dir = code_dir.parent / "data"

# paths
DEFAULT_IMG_PATH = data_dir / "example_data/example_science_image_mef.fits"
DEFAULT_SAVE_DIR = data_dir / "example_data"
DEFAULT_CORRECTION_DIR = "/home/winter/GIT/winter_linearity/data/linearity_corrections" + zenodo_version

example_data_dir = data_dir / "example_data"

EXAMPLE_IMG_PATH = example_data_dir / "example_science_image_mef.fits"
EXAMPLE_CORRECTED_IMG_PATH = example_data_dir / "corrected_example_science_image_mef.fits"
EXAMPLE_MASKED_IMG_PATH = example_data_dir / "masked_example_science_image_mef.fits"
_corrections_dir = os.getenv("WINTERNLC_DIR")
if _corrections_dir is None:
    corrections_dir = Path.home() / "Data/winternlc/"
    logger.warning(f"No data directory set, using {corrections_dir}")
else:
    corrections_dir = Path(_corrections_dir)

corrections_dir.mkdir(parents=True, exist_ok=True)

test_directory = "/data/flats_iwr/20240610"
output_directory = data_dir / "linearity_corrections" + zenodo_version

# variables
DEFAULT_CUTOFF = 56000
