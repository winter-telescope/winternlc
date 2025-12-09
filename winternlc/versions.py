"""
Central script for assigning nlc versions based on image headers
"""

from astropy.io import fits

from winternlc.zenodo import LATEST_ZENODO_VERSION


def get_nlc_version(header: fits.header) -> str:
    """
    Get the appropriate WinterNLC version based on the image header.

    :param header: FITS header of the image
    :return: Version string of the WinterNLC corrections
    """
    # The group decided to move to v2.0 as default on 2025-06-11
    version = LATEST_ZENODO_VERSION
    return version
