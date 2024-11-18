"""
Central script for assigning nlc versions based on image headers
"""

from astropy.io import fits

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
    if board_id == 4:
        fallback_version = "v0.1"

    # Logic for choosing version based on dates/firmware etc
    version = fallback_version

    return version
