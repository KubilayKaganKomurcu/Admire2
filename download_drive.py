#!/usr/bin/env python3
import os
import re
import requests
from bs4 import BeautifulSoup
import gdown

def get_folder_contents(folder_id):
    """Get folder contents by parsing Google Drive page."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    try:
        # Use gdown's session for cookies
        sess = requests.Session()
        res = sess.get(url, headers={"User-Agent": "Mozilla/5.0"})
        
        # Find all file/folder IDs in the page
        # Pattern for Google Drive IDs
        pattern = r'\["([a-zA-Z0-9_-]{25,})","([^"]+)"'
        matches = re.findall(pattern, res.text)
        
        items = []
        seen = set()
        for file_id, name in matches:
            if file_id not in seen and len(file_id) > 20:
                seen.add(file_id)
                # Skip common non-file IDs
                if name and not name.startswith('http') and '.' in name or len(name) < 50:
                    items.append((file_id, name))
        
        return items
    except Exception as e:
        print(f"Error: {e}")
        return []

def download_with_gdown(folder_id, output_dir):
    """Try to download using gdown's folder download with remaining-ok."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url, output=output_dir, quiet=False, remaining_ok=True)
        return True
    except Exception as e:
        print(f"gdown folder failed: {e}")
        return False

def download_languages(main_folder_id, output_dir):
    """Download each language folder separately."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Known language folder IDs from the Google Drive
    # These are the subfolders in the main folder
    languages = [
        "Chinese", "Georgian", "Greek", "Igbo", "Kazakh",
        "Norwegian", "Portuguese-Brazil", "Portuguese-Portugal",
        "Russian", "Serbian", "Slovak", "Slovenian",
        "Spanish-Ecuador", "Turkish", "Uzbek"
    ]
    
    print(f"Attempting to download from main folder: {main_folder_id}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    # First try downloading each language folder with gdown
    main_url = f"https://drive.google.com/drive/folders/{main_folder_id}"
    
    # Download the main folder - gdown will create subfolders
    print("Downloading main folder (this may take a while)...")
    try:
        gdown.download_folder(
            main_url, 
            output=output_dir, 
            quiet=False, 
            remaining_ok=True,
            use_cookies=False
        )
    except Exception as e:
        print(f"Note: {e}")
        print("Continuing anyway...")
    
    print("=" * 50)
    print("Download attempt complete!")
    print(f"Check {output_dir} for downloaded files.")

if __name__ == "__main__":
    FOLDER_ID = "1kVC3a1ZqmYf6O5PyQ6NQuv_orgzICUOg"
    OUTPUT_DIR = "./data/images"
    download_languages(FOLDER_ID, OUTPUT_DIR)
