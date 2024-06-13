from winternlc.config import cor_dir, cutoff, fits_file, save_dir
from winternlc.example import test_mask, test_nonlinearity

if __name__ == "__main__":
    test_nonlinearity(fits_file, cor_dir, save_dir, cutoff)
    test_mask(fits_file, cor_dir, save_dir, cutoff)
