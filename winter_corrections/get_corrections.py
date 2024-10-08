import os
import subprocess
from config import zenodo_url, cor_dir  # Importing the URL and cor_dir from config

# Function to create the correction folder (cor_dir) if it doesn't exist
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Function to download only .npy files from Zenodo using wget into cor_dir
def download_files(zenodo_url, download_dir):
    # Use wget to download only .npy files and handle the content-disposition properly
    command = f'wget -r -np -nH --cut-dirs=100 --content-disposition -A "*.npy" -P {download_dir} {zenodo_url}'
    subprocess.run(command, shell=True, check=True)

# Main function
def main():
    # Create the folder for corrections (cor_dir)
    download_dir = create_folder(cor_dir)
    
    # Download only .npy files to the correction folder (cor_dir)
    download_files(zenodo_url, download_dir)
    print(f"Only .npy files downloaded to the folder: {download_dir}")

if __name__ == "__main__":
    main()
