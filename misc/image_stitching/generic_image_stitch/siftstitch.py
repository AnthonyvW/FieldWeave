#!/usr/bin/env python3
"""
Image Stitching Script
Stitches together overlapping JPEG images from a folder into a single panorama.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def load_images_from_folder(folder_path):
    """Load all JPEG images from the specified folder."""
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    
    # Get all image files and sort them
    for file in sorted(os.listdir(folder_path)):
        if any(file.endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        raise ValueError(f"No JPEG images found in {folder_path}")
    
    print(f"Found {len(image_files)} images:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    # Load images
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not load {img_file}, skipping...")
            continue
        images.append(img)
    
    return images


def stitch_images(images):
    """
    Stitch images together using OpenCV's Stitcher.
    
    Args:
        images: List of images to stitch
    
    Returns:
        Stitched image or None if stitching failed
    """
    print(f"\nStitching {len(images)} images...")
    
    # Create stitcher object
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    
    # Perform stitching
    status, stitched = stitcher.stitch(images)
    
    # Check stitching status
    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        return stitched
    else:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
        }
        error_msg = error_messages.get(status, f"Unknown error (code: {status})")
        print(f"Stitching failed: {error_msg}")
        return None


def crop_black_borders(image):
    """Remove black borders from stitched image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-black regions
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description='Stitch overlapping JPEG images into a panorama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_stitcher.py /path/to/images
  python image_stitcher.py /path/to/images --output my_panorama.jpg
  python image_stitcher.py /path/to/images --no-crop
        """
    )
    
    parser.add_argument('input_folder', type=str,
                       help='Folder containing JPEG images to stitch')
    parser.add_argument('--output', '-o', type=str, default='stitched_panorama.jpg',
                       help='Output filename (default: stitched_panorama.jpg)')
    parser.add_argument('--no-crop', action='store_true',
                       help='Skip automatic cropping of black borders')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return 1
    
    try:
        # Load images
        images = load_images_from_folder(args.input_folder)
        
        if len(images) < 2:
            print("Error: Need at least 2 images to stitch")
            return 1
        
        # Stitch images
        result = stitch_images(images)
        
        if result is None:
            print("\nStitching failed. Tips:")
            print("  - Ensure images have sufficient overlap (30-50%)")
            print("  - Images should be taken from the same position")
            print("  - Ensure images are in the correct order")
            return 1
        
        # Crop black borders unless disabled
        if not args.no_crop:
            print("Cropping black borders...")
            result = crop_black_borders(result)
        
        # Save result
        cv2.imwrite(args.output, result)
        print(f"\nPanorama saved to: {args.output}")
        print(f"Output size: {result.shape[1]}x{result.shape[0]} pixels")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())