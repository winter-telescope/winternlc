from winternlc.config import cor_dir, DEFAULT_CUTOFF, DEFAULT_IMG_PATH, save_dir
from winternlc.example import apply_mask_mef, apply_nlc_mef

if __name__ == "__main__":
    apply_nlc_mef(DEFAULT_IMG_PATH, cor_dir, save_dir, DEFAULT_CUTOFF)
    apply_mask_mef(DEFAULT_IMG_PATH, cor_dir, save_dir, DEFAULT_CUTOFF)
