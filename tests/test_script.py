from winternlc.apply import apply_mask_mef, apply_nlc_mef
from winternlc.config import (
    DEFAULT_CORRECTION_DIR,
    DEFAULT_CUTOFF,
    DEFAULT_IMG_PATH,
    DEFAULT_SAVE_DIR,
)

if __name__ == "__main__":
    apply_nlc_mef(
        DEFAULT_IMG_PATH, DEFAULT_CORRECTION_DIR, DEFAULT_SAVE_DIR, DEFAULT_CUTOFF
    )
    apply_mask_mef(
        DEFAULT_IMG_PATH, DEFAULT_CORRECTION_DIR, DEFAULT_SAVE_DIR, DEFAULT_CUTOFF
    )
