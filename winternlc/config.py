import logging
import os
from pathlib import Path

from winternlc.zenodo import LATEST_ZENODO_VERSION, ZENODO_URL_MAP

logger = logging.getLogger(__name__)

code_dir = Path(__file__).parent

data_dir = code_dir.parent / "data"

example_data_dir = data_dir / "example_data"

EXAMPLE_IMG_PATH = example_data_dir / "example_science_image_mef.fits"
EXAMPLE_CORRECTED_IMG_PATH = (
    example_data_dir / "corrected_example_science_image_mef.fits"
)
EXAMPLE_MASKED_IMG_PATH = example_data_dir / "masked_example_science_image_mef.fits"

_corrections_dir = os.getenv("WINTERNLC_DIR")
if _corrections_dir is None:
    base_corrections_dir = Path.home() / "Data/winternlc/"
    if not base_corrections_dir.exists():
        base_corrections_dir.mkdir(parents=True)
        logger.warning(f"No data directory set, using {base_corrections_dir}")
else:
    base_corrections_dir = Path(_corrections_dir)


def get_correction_dir(version: str = LATEST_ZENODO_VERSION) -> Path:
    """
    Returns the path to the correction data directory
    for a given version of the WinterNLC corrections

    :param version: Version of the WinterNLC corrections

    :return: Path to the correction data directory
    """

    assert (
        version in ZENODO_URL_MAP
    ), f"Version {version} is not available. Please choose from {ZENODO_URL_MAP.keys()}"

    corrections_dir = base_corrections_dir / version
    corrections_dir.mkdir(parents=True, exist_ok=True)

    return corrections_dir


test_directory = "/data/flats_iwr/20240610"
output_directory = data_dir / f"linearity_corrections/{LATEST_ZENODO_VERSION}"

# variables
DEFAULT_CUTOFF = 56000
