"""
Central script for assigning nlc versions based on image headers
"""

from datetime import datetime

from astropy.io import fits

from winternlc.config import VERSION_DATES
from winternlc.zenodo import LATEST_ZENODO_VERSION


def get_nlc_version(header: fits.header) -> str:
    # The group decided to move to v2.0 as default on 2025-06-11
    version = LATEST_ZENODO_VERSION
    return version
