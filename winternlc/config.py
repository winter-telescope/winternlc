import logging
import os
from datetime import datetime
from pathlib import Path

from .zenodo import LATEST_ZENODO_VERSION, ZENODO_URL_MAP

logger = logging.getLogger(__name__)

code_dir = Path(__file__).parent

data_dir = code_dir.parent / "data"

example_data_dir = data_dir / "example_data"

available_versions = ["v0.1", "v1.0", "v1.1", "v2.0"]

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

    available_versions = ["v0.1", "v1.0", "v1.1", "v2.0"]
    assert (
        version in available_versions
    ), f"Version {version} is not available. Please choose from {available_versions}"

    corrections_dir = base_corrections_dir / version
    corrections_dir.mkdir(parents=True, exist_ok=True)

    return corrections_dir


test_directory = "/data/flats_iwr/20240610"
output_directory = data_dir / f"linearity_corrections/{LATEST_ZENODO_VERSION}"

# variables and dates
DEFAULT_CUTOFF = 56000
CROSSOVER_POINT = 25_000
ORDER_LOWER = 11
ORDER_UPPER = 5

VERSION_DATES = {
    "v0.1": datetime(2024, 6, 1),
    "v1.0": datetime(2024, 8, 3),
    "v1.1": datetime(2024, 8, 4),
    "v2.0": datetime(2025, 2, 1),
}
