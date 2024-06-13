from winternlc.example import test_nonlinearity, test_mask
from winternlc.config import fits_file, cor_dir, save_dir, cutoff

if __name__ == "__main__":
    test_nonlinearity(fits_file, cor_dir, save_dir, cutoff)
    test_mask(fits_file, cor_dir, save_dir, cutoff)
