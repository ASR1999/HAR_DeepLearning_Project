# download_dataset.py
import requests
import zipfile
import io
import os

# URL for the dataset
DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
# Path to the directory where data should be saved
DATA_DIR = "./data/"

def download_and_extract():
    """
    Downloads and extracts the UCI HAR Dataset.
    """
    print(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        # Create the data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Send a GET request to the URL
        response = requests.get(DATASET_URL)
        response.raise_for_status()  # Check for request errors
        
        # Create a ZipFile object from the in-memory content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"Extracting dataset to {DATA_DIR}...")
            z.extractall(DATA_DIR)
        
        print("Download and extraction complete.")
        print(f"Data is available in '{DATA_DIR}UCI HAR Dataset/'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_and_extract()