#!/usr/bin/env python3
"""
Download Turkish folder from Google Drive.
Structure: Turkish/ -> phrase folders -> 5 PNGs each
"""
import os
import gdown

def download_turkish():
    """Download only the Turkish folder."""
    
    # Main folder containing all languages
    MAIN_FOLDER_ID = "1kVC3a1ZqmYf6O5PyQ6NQuv_orgzICUOg"
    OUTPUT_DIR = "./data/images"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Downloading Turkish folder...")
    print(f"Output: {OUTPUT_DIR}/Turkish/")
    print("=" * 50)
    
    # First, get the Turkish folder ID by downloading folder list
    main_url = f"https://drive.google.com/drive/folders/{MAIN_FOLDER_ID}"
    
    try:
        # Download just Turkish - gdown will detect subfolders
        gdown.download_folder(
            main_url,
            output=OUTPUT_DIR,
            quiet=False,
            remaining_ok=True,
            use_cookies=False
        )
    except Exception as e:
        print(f"Error: {e}")
    
    # Check if Turkish folder exists
    turkish_path = os.path.join(OUTPUT_DIR, "Turkish")
    if os.path.exists(turkish_path):
        phrases = os.listdir(turkish_path)
        print(f"\n✓ Turkish folder downloaded with {len(phrases)} phrases")
        for p in phrases[:5]:
            phrase_path = os.path.join(turkish_path, p)
            if os.path.isdir(phrase_path):
                files = os.listdir(phrase_path)
                print(f"  - {p}: {len(files)} files")
    else:
        print("\n✗ Turkish folder not found. Trying alternative method...")
        
        # Alternative: Try to find and download Turkish folder specifically
        # You may need to provide the Turkish folder ID directly
        print("\nTo download Turkish only, find the folder ID from the browser:")
        print("1. Open: https://drive.google.com/drive/folders/1kVC3a1ZqmYf6O5PyQ6NQuv_orgzICUOg")
        print("2. Click on 'Turkish' folder")  
        print("3. Copy the folder ID from URL")
        print("4. Run: gdown --folder https://drive.google.com/drive/folders/TURKISH_FOLDER_ID -O ./data/images/Turkish")

if __name__ == "__main__":
    download_turkish()
