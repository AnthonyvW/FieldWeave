#!/usr/bin/env python3
"""
Image Converter Script
Converts PNG and TIFF files to JPG format and saves them in a 'converted' subfolder.

Usage:
    python image_converter.py <folder_path>
"""

import os
import sys
from pathlib import Path
from PIL import Image


def convert_to_jpg(input_path, output_folder, quality=95):
    """
    Convert an image file to JPG format.
    
    Args:
        input_path: Path to the input image file
        output_folder: Path to the output folder
        quality: JPG quality (1-100, default 95)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert RGBA to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Generate output filename
            input_file = Path(input_path)
            output_filename = input_file.stem + '.jpg'
            output_path = output_folder / output_filename
            
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            print(f"Converted: {input_file.name} -> {output_filename}")
            return True
            
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def main():
    """Main function to process all PNG and TIFF files in the specified directory."""
    
    # Check if folder argument is provided
    if len(sys.argv) != 2:
        print("Usage: python image_converter.py <folder_path>")
        print("\nExample:")
        print("  python image_converter.py /path/to/images")
        print("  python image_converter.py ./my_images")
        sys.exit(1)
    
    # Get the folder path from argument
    input_folder = Path(sys.argv[1])
    
    # Validate the folder exists
    if not input_folder.exists():
        print(f"Error: Folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        sys.exit(1)
    
    print(f"Input folder: {input_folder.absolute()}")
    
    # Create output folder inside the input folder
    output_folder = input_folder / 'converted'
    output_folder.mkdir(exist_ok=True)
    print(f"Output folder: {output_folder.absolute()}\n")
    
    # Supported extensions
    supported_extensions = {'.png', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in supported_extensions:
        image_files.extend(input_folder.glob(f'*{ext}'))
        image_files.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print("No PNG or TIFF files found in the specified directory.")
        return
    
    print(f"Found {len(image_files)} image(s) to convert.\n")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for image_file in image_files:
        if convert_to_jpg(image_file, output_folder):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output location: {output_folder.absolute()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()