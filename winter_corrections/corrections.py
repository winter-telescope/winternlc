import os
import numpy as np


def rational_func(x, a0, a1, a2, a3, b0, b1, b2, b3):
    return (a0 + a1*x + a2*x**2 + a3*x**3) / (1 + b0*x + b1*x**2 + b2*x**3 + b3*x**4)

def fitted_func(x, coeffs):
    return rational_func(x, *coeffs)

def nonlinearity_correction(image, board_id, ext, cor_dir, cutoff):
    """
    Applies nonlinearity correction to an image using precomputed rational coefficients.
    """
    rat_coeffs_path = os.path.join(cor_dir, f'rat_coeffs_board_{board_id}_ext_{ext-1}.npy')
    
    if os.path.exists(rat_coeffs_path):
        rat_coeffs = np.load(rat_coeffs_path)
        
        # Apply cutoff
        image = np.clip(image, None, cutoff)
        
        # Normalize image by cutoff
        image = image / cutoff
        
        # Vectorized application of the fitted function
        rat_coeffs = rat_coeffs.reshape(-1, 8)
        image = fitted_func(image.flatten(), rat_coeffs.T).reshape(image.shape)
        
        # Scale back by cutoff
        image = cutoff * image
        return image
    else:
        raise FileNotFoundError(f'Rational coefficients file not found for board_id {board_id}')
        
def mask_bad_pixels(image, board_id, ext, cor_dir):
    """
    Applies a bad pixel mask to an image.
    """
    mask_path = os.path.join(cor_dir, f'bad_pixel_mask_board_{board_id}_ext_{ext-1}.npy')
    
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        image[mask] = np.nan  # Set bad pixels to NaN
        return image
    else:
        raise FileNotFoundError(f'Bad pixel mask file not found for board_id {board_id}')