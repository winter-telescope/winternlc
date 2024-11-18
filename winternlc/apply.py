"""
Apply non-linearity correction to multi-extension FITS file(s)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from winternlc.config import DEFAULT_CUTOFF
from winternlc.get_corrections import check_for_files
from winternlc.mask import apply_mask_single
from winternlc.non_linear_correction import apply_nlc_single

logger = logging.getLogger(__name__)


def apply_nlc_mef(
    fits_file: Path,
    save_dir: Path | str | None = None,
    version: str | None = None,
    cutoff: float = DEFAULT_CUTOFF,
    output_suffix: str | None = "corrected_",
):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each
    extension, and write the corrected FITS file to disk.

    :param fits_file: Path to the FITS file
    :param save_dir: Directory to save the corrected FITS file
    :param version: Version of the WinterNLC corrections
    :param cutoff: Cutoff value for the correction
    :param output_suffix: Suffix to append to the output file name

    :return: None
    """
    check_for_files()

    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            corrected_image = apply_nlc_single(image, header, version, cutoff)
            hdul[ext].data = corrected_image

        if save_dir is None:
            save_dir = Path(fits_file).parent

        corrected_fits_file = save_dir / f"{output_suffix}{fits_file.name}"
        hdul.writeto(corrected_fits_file, overwrite=True)
        logger.info(f"Corrected FITS file saved to {corrected_fits_file}")


def apply_mask_mef(
    fits_file: str | Path,
    save_dir: Path | str | None = None,
    version: str | None = None,
    output_suffix: str | None = "masked_",
):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each
    extension, and write the corrected FITS file to disk.

    :param fits_file: Path to the FITS file
    :param save_dir: Directory to save the corrected FITS file
    :param version: Version of the WinterNLC corrections, default is None
    :param output_suffix: Suffix to append to the output file name

    :return: None
    """
    check_for_files()

    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            corrected_image = apply_mask_single(image, header, version)
            hdul[ext].data = corrected_image

        if save_dir is None:
            save_dir = Path(fits_file).parent

        corrected_fits_file = save_dir / f"{output_suffix}{fits_file.name}"
        hdul.writeto(corrected_fits_file, overwrite=True)
        logger.info(f"Corrected FITS file saved to {corrected_fits_file}")


def nlc_cli():
    """
    Command-line interface for applying non-linearity
    correction to multi-extension FITS file(s)
    """
    parser = argparse.ArgumentParser(
        description="Apply non-linearity correction to multi-extension FITS file"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Apply bad pixel mask instead of NLC",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        type=str,
        help="Directory to save the corrected FITS file(s) "
        "(default: same as input file)",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        type=str,
        help="Version of the WinterNLC corrections to use",
    )
    parser.add_argument("files", nargs="+", help="FITS file(s) to correct")

    args = parser.parse_args()

    logger.info("Applying non-linearity correction to multi-extension FITS file")

    file_paths = []

    for f_name in args.files:
        path = Path(f_name)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {path}")
        elif path.is_dir():
            file_paths.extend(list(path.glob("*.fits")))
        else:
            file_paths.append(path)

    for path in file_paths:
        apply_nlc_mef(
            fits_file=path,
            save_dir=args.output_dir,
        )

    if args.mask:
        for path in file_paths:
            apply_mask_mef(
                fits_file=path,
                save_dir=args.output_dir,
            )


if __name__ == "__main__":
    nlc_cli()
