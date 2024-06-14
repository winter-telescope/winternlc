<<<<<<< HEAD
# versions
available_versions = ["v0.1" , "v1.0", "v1.1"]
urls = ["https://zenodo.org/records/13905735", "https://zenodo.org/records/13863497", "https://zenodo.org/records/13905772"]
zenodo_version = "v1.1"  

if zenodo_version in available_versions:
    index = available_versions.index(zenodo_version)
    zenodo_url = urls[index]
else:
    raise ValueError(f"Version {zenodo_version} is not available. Please choose from {available_versions}")

from pathlib import Path

code_dir = Path(__file__).parent

data_dir = code_dir / "data"

# paths
DEFAULT_IMG_PATH = data_dir / "example_data/example_science_image_mef.fits"
save_dir = data_dir / "example_data"
cor_dir = "/home/winter/GIT/winter_linearity/data/linearity_corrections" + zenodo_version
test_directory = "/data/flats_iwr/20240610"
output_directory = data_dir / "linearity_corrections" + zenodo_version

# variables
DEFAULT_CUTOFF = 56000
