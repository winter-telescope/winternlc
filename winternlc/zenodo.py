"""
This module contains the Zenodo version and URL mapping for the WinterNLC project.
"""
import os

CURRENT_ZENODO_VERSION = "v1.1"

ZENODO_VERSION = os.getenv("WINTERNLC_VERSION", CURRENT_ZENODO_VERSION)

_ZENODO_URL_MAP = {
    "v0.1": "https://zenodo.org/record/13905735",
    "v1.0": "https://zenodo.org/record/13863497",
    "v1.1": "https://zenodo.org/api/records/13905772/files-archive",
}

assert ZENODO_VERSION in _ZENODO_URL_MAP, f"Version {ZENODO_VERSION} is not available. Please choose from {_ZENODO_URL_MAP.keys()}"

ZENODO_URL = _ZENODO_URL_MAP[ZENODO_VERSION]
