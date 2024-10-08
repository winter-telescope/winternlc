import os
import subprocess
from config import zenodo_url, cor_dir 

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def download_files(zenodo_url, download_dir):
    command = f'wget -r -np -nH --cut-dirs=100 --content-disposition -A "*.npy" -P {download_dir} {zenodo_url}'
    subprocess.run(command, shell=True, check=True)

def main():
    download_dir = create_folder(cor_dir)
    download_files(zenodo_url, download_dir)
    
if __name__ == "__main__":
    main()
