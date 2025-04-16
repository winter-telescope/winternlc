"""
Central script for assigning nlc versions based on image headers
"""

from datetime import datetime

from astropy.io import fits

from winternlc.config import VERSION_DATES
from winternlc.zenodo import LATEST_ZENODO_VERSION


def get_nlc_version(header: fits.header) -> str:
    """
    Assigns the version of the WinterNLC corrections based on the image header.

    :param header: Image header

    :return: Version of the WinterNLC corrections
    """
    # Choose the fallback version based on the BOARD_ID
    fallback_version = LATEST_ZENODO_VERSION
    board_id = header["BOARD_ID"]

    try:
        # Get image observation date
        obs_date = datetime.strptime(
            header["DATE-OBS"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        # Select the latest version valid for the observation date
        version = max(
            (v for v, d in VERSION_DATES.items() if d <= obs_date),
            default=fallback_version,
            key=VERSION_DATES.get,
        )
    except:
        version = fallback_version

    # hacky overwrite, since the v0.1 version is not available for board_id 4
    # remove with new corrections
    if board_id == 4 or board_id == 6 or board_id == 1:
        version = "v0.1"

    return version
