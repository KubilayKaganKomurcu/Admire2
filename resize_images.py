"""
Image Preprocessing Script
==========================
Resize all images in the data folder to reduce file sizes.

Usage:
    python resize_images.py
"""

import os
from pathlib import Path
from PIL import Image
import shutil

def resize_images(
    data_dir: str = "data/languages",
    max_size: int = 512,
    quality: int = 85,
    backup: bool = True
):
    """
    Resize all images in the data directory.
    
    Args:
        data_dir: Directory containing language folders with images
        max_size: Maximum width/height in pixels
        quality: JPEG quality (1-100)
        backup: If True, create backup of original images first
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f"*{ext}"))
        image_files.extend(data_path.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    if not image_files:
        return
    
    # Calculate total original size
    original_total = sum(f.stat().st_size for f in image_files)
    print(f"Original total size: {original_total / 1024 / 1024:.2f} MB")
    
    # Backup original images if requested
    if backup:
        backup_dir = Path("data/images_backup")
        if not backup_dir.exists():
            print(f"\nCreating backup at {backup_dir}...")
            backup_dir.mkdir(parents=True, exist_ok=True)
            for img_path in image_files:
                rel_path = img_path.relative_to(data_path)
                backup_path = backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, backup_path)
            print(f"Backup complete!")
        else:
            print(f"Backup already exists at {backup_dir}, skipping backup")
    
    # Resize images
    print(f"\nResizing images to max {max_size}x{max_size}...")
    resized_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            with Image.open(img_path) as img:
                original_size = img.size
                
                # Check if resize needed
                if img.width <= max_size and img.height <= max_size:
                    skipped_count += 1
                    continue
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate new size maintaining aspect ratio
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                
                # Resize
                img_resized = img.resize(new_size, Image.LANCZOS)
                
                # Save as JPEG (smaller) - change extension
                new_path = img_path.with_suffix('.jpg')
                img_resized.save(new_path, format='JPEG', quality=quality, optimize=True)
                
                # Remove original if it was a different format
                if new_path != img_path:
                    img_path.unlink()
                
                resized_count += 1
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(image_files)}...")
                    
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            error_count += 1
    
    # Calculate new total size
    new_image_files = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.jpeg"))
    new_total = sum(f.stat().st_size for f in new_image_files if f.exists())
    
    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"  Resized: {resized_count}")
    print(f"  Skipped (already small): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"\n  Original size: {original_total / 1024 / 1024:.2f} MB")
    print(f"  New size: {new_total / 1024 / 1024:.2f} MB")
    print(f"  Saved: {(original_total - new_total) / 1024 / 1024:.2f} MB ({100 * (1 - new_total/original_total):.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resize images in data folder")
    parser.add_argument("--dir", default="data/languages", help="Directory containing images")
    parser.add_argument("--size", type=int, default=512, help="Max width/height in pixels")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of original images")
    
    args = parser.parse_args()
    
    resize_images(
        data_dir=args.dir,
        max_size=args.size,
        quality=args.quality,
        backup=not args.no_backup
    )

