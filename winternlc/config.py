from pathlib import Path
import logging
import os
from winternlc.zenodo import CURRENT_ZENODO_VERSION

logger = logging.getLogger(__name__)

code_dir = Path(__file__).parent

data_dir = code_dir.parent / "data"

example_data_dir = data_dir / "example_data"

EXAMPLE_IMG_PATH = example_data_dir / "example_science_image_mef.fits"
EXAMPLE_CORRECTED_IMG_PATH = example_data_dir / "corrected_example_science_image_mef.fits"
EXAMPLE_MASKED_IMG_PATH = example_data_dir / "masked_example_science_image_mef.fits"

_corrections_dir = os.getenv("WINTERNLC_DIR")
if _corrections_dir is None:
    base_corrections_dir = Path.home() / "Data/winternlc/"
    logger.warning(f"No correction data directory set, using {base_corrections_dir}")
else:
    base_corrections_dir = Path(_corrections_dir)

corrections_dir = base_corrections_dir / CURRENT_ZENODO_VERSION
corrections_dir.mkdir(parents=True, exist_ok=True)

test_directory = "/data/flats_iwr/20240610"
output_directory = data_dir / f"linearity_corrections/{CURRENT_ZENODO_VERSION}"

# variables
DEFAULT_CUTOFF = 56000
