#!/usr/bin/env python3
"""
Script to download nested Google Drive folders with >50 files.
Structure: Main Folder -> Languages -> Phrases -> 5 PNGs each
"""

import os
import gdown
from gdown.download_folder import _get_folder_list


def get_subfolders(folder_id):
    """Get list of subfolders from a Google Drive folder."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        return_code, gdrive_files = _get_folder_list(url)
        if return_code != 0:
            print(f"Warning: Could not fully retrieve folder {folder_id}")
        
        # Filter for folders only (folders have mimeType containing 'folder')
        subfolders = []
        files = []
        for f in gdrive_files:
            # gdown returns tuples of (id, name, mimeType, ...)
            if len(f) >= 3:
                file_id, name, mime = f[0], f[1], f[2] if len(f) > 2 else ""
                if "folder" in str(mime).lower():
                    subfolders.append((file_id, name))
                else:
                    files.append((file_id, name))
            else:
                file_id, name = f[0], f[1]
                subfolders.append((file_id, name))  # Assume folder if no mime
        
        return subfolders, files
    except Exception as e:
        print(f"Error getting folder list: {e}")
        return [], []


def download_folder_recursive(folder_id, output_dir, depth=0):
    """Recursively download folders, handling >50 file limit by going deeper."""
    indent = "  " * depth
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{indent}Processing folder: {output_dir}")
    
    subfolders, files = get_subfolders(folder_id)
    
    # Download files in current folder
    if files:
        print(f"{indent}  Found {len(files)} files")
        for file_id, file_name in files:
            output_path = os.path.join(output_dir, file_name)
            if os.path.exists(output_path):
                print(f"{indent}  Skipping (exists): {file_name}")
                continue
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output_path, quiet=True)
                print(f"{indent}  Downloaded: {file_name}")
            except Exception as e:
                print(f"{indent}  Error downloading {file_name}: {e}")
    
    # Recursively process subfolders
    if subfolders:
        print(f"{indent}  Found {len(subfolders)} subfolders")
        for subfolder_id, subfolder_name in subfolders:
            subfolder_path = os.path.join(output_dir, subfolder_name)
            download_folder_recursive(subfolder_id, subfolder_path, depth + 1)


def main():
    # The folder ID from your URL
    # https://drive.google.com/drive/folders/1kVC3a1ZqmYf6O5PyQ6NQuv_orgzICUOg
    FOLDER_ID = "1kVC3a1ZqmYf6O5PyQ6NQuv_orgzICUOg"
    
    # Output directory
    OUTPUT_DIR = "./images"  # Change this to your desired output path
    
    print(f"Starting download from folder: {FOLDER_ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    
    download_folder_recursive(FOLDER_ID, OUTPUT_DIR)
    
    print("=" * 50)
    print("Download complete!")


if __name__ == "__main__":
    main()

