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

    return LATEST_ZENODO_VERSION
