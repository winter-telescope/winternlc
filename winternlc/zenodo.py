"""
This module contains the Zenodo version and URL mapping for the WinterNLC project.
"""

LATEST_ZENODO_VERSION = "v1.1"

ZENODO_URL_MAP = {
    "v0.1": "https://zenodo.org/api/records/13905735/files-archive",
    "v1.0": "https://zenodo.org/api/records/13863497/files-archive",
    "v1.1": "https://zenodo.org/api/records/13905772/files-archive",
}


def get_zenodo_url(version: str = LATEST_ZENODO_VERSION) -> str:
    """
    Returns the Zenodo URL for a given version of the WinterNLC project.

    :param version: Version of the WinterNLC project

    :return: Zenodo URL
    """

    assert (
        version in ZENODO_URL_MAP
    ), f"Version {version} is not available. Please choose from {ZENODO_URL_MAP.keys()}"

    return ZENODO_URL_MAP[version]
