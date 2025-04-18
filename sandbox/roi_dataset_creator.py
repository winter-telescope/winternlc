# This is a script to make ROI datasets based on the full dataset

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from utils import plot_image
from winter_image import WinterImage

# path to the data
data_dir = os.path.join(os.getenv("HOME"), "data", "nlc_data")

# the data has names like "median_exp_2.15.fits", and is a 6 layer multi-extension fits file

# load all the data in the directory
data_files = glob.glob(os.path.join(data_dir, "median_exp_*.fits"))

plot_data = False
# load and plot a random file
for data_file in data_files:
    filename = os.path.basename(data_file)
    print(f"loading {filename}")
    img = WinterImage(data_file)

    # plot the image mosaic
    # img.plot_mosaic(single_image_width=2, channel_labels="board_id")

    # try to plot a single layer
    # does the header work?
    addrs = ["pc", "sb"]
    for addr in addrs:
        data = img.imgs[addr]
        header = img.headers[addr]

        # full image
        if plot_data:
            plot_image(
                data, title=f"{addr}-Full Frame: Exposure Time = {header['EXP_ACT']} s"
            )

        # select an ROI
        wy, wx = data.shape
        print(f"the data shape is: (wx, wy) = {wx, wy}")
        w_roi = 512
        x_min = 0
        x_max = x_min + w_roi
        y_min = wy - w_roi
        y_max = wy

        print(
            f"the ROI is: (x_min, x_max, y_min, y_max) = {x_min, x_max, y_min, y_max}"
        )
        roi_data = data[y_min:y_max, x_min:x_max]
        print(f"the roi shape is: {roi_data.shape}")
        if plot_data:
            plot_image(
                roi_data, title=f"{addr}-ROI: Exposure Time = {header['EXP_ACT']} s"
            )

        # save the ROI to a new fits file
        roi_directory = os.path.join(data_dir, "roi", addr)
        os.makedirs(roi_directory, exist_ok=True)

        # save the ROI to a new fits file
        roi_filename = f"{addr}_roi_{filename}"
        roi_filepath = os.path.join(roi_directory, roi_filename)
        print(f"saving {roi_filename}")
        hdu = fits.PrimaryHDU(roi_data, header=header)
        hdu.writeto(roi_filepath, overwrite=True)
if plot_data:
    plt.show()
