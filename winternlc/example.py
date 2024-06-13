import os
from astropy.io import fits
import time
from corrections import nonlinearity_correction, mask_bad_pixels
from config import fits_file, cor_dir, save_dir, cutoff

def test_nonlinearity(fits_file, cor_dir, save_dir, cutoff):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each extension.
    """
    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            board_id = header.get('BOARD_ID', None)
            if board_id is not None:
                print(f"Processing extension {ext} with BOARD_ID {board_id}")
                start = time.time()
                corrected_image = nonlinearity_correction(image, board_id, ext, cor_dir, cutoff)
                end = time.time()
                print(f"took {end-start} s to execute")
                hdul[ext].data = corrected_image
            else:
                print(f"Skipping extension {ext} as it does not have a BOARD_ID")
        
        corrected_fits_file = os.path.join(save_dir, 'corrected_' + os.path.basename(fits_file))
        hdul.writeto(corrected_fits_file, overwrite=True)
        print(f"Corrected FITS file saved to {corrected_fits_file}")
        
def test_mask(fits_file, cor_dir, save_dir, cutoff):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each extension.
    """
    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            board_id = header.get('BOARD_ID', None)
            if board_id is not None:
                print(f"Masking extension {ext} with BOARD_ID {board_id}")
                start = time.time()
                corrected_image = mask_bad_pixels(image, board_id, ext, cor_dir)
                end = time.time()
                print(f"took {end-start} s to execute")
                hdul[ext].data = corrected_image
            else:
                print(f"Skipping extension {ext} as it does not have a BOARD_ID")
        
        corrected_fits_file = os.path.join(save_dir, 'masked_' + os.path.basename(fits_file))
        hdul.writeto(corrected_fits_file, overwrite=True)
        print(f"Corrected FITS file saved to {corrected_fits_file}")

if __name__ == "__main__":
    test_nonlinearity(fits_file, cor_dir, save_dir, cutoff)
    test_mask(fits_file, cor_dir, save_dir, cutoff)
