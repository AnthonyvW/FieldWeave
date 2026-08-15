import os
from pathlib import Path
from stitching import AffineStitcher
import cv2 as cv
import glob
import re
from typing import List, Tuple, Optional
import statistics

def extract_f_score(filename: str) -> Optional[float]:
    """
    Extract F score from filename in format "Z{score} F{score}.extension"
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        Optional[float]: The F score if found, None otherwise
    """
    try:
        # Look for pattern "F{score}" in filename
        match = re.search(r'F([\d.]+)', filename)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass
    return None

def get_images_with_f_scores(folder_path: Path) -> List[Tuple[Path, float]]:
    """
    Get all images in folder with their F scores.
    
    Args:
        folder_path (Path): Path to the folder containing images
        
    Returns:
        List[Tuple[Path, float]]: List of (image_path, f_score) tuples
    """
    images_with_scores = []
    
    for image_path in folder_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            f_score = extract_f_score(image_path.name)
            if f_score is not None:
                images_with_scores.append((image_path, f_score))
            else:
                print(f"Warning: Could not extract F score from {image_path.name}")
    
    return images_with_scores

def execute_focus_stack(input_files: List[Path], output_file: Path, description: str = "") -> bool:
    """
    Execute the focus stacking command using the external focus-stack.exe tool.
    
    Args:
        input_files (List[Path]): List of input image files to stack
        output_file (Path): Path for the output stacked image
        description (str): Description for logging purposes
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert paths to strings and quote them
    input_args = ' '.join([f'"{str(file)}"' for file in input_files])
    
    # Construct the command
    command = (
        f".\\focus-stack\\focus-stack.exe "
        f"{input_args} "
        f"--output=\"{output_file}\" "
        f"--consistency=0 "
        f"--no-whitebalance "
        f"--no-contrast"
    )
    
    print(f"Processing: {description}")
    print(f"Running command: {command}")
    
    try:
        # Execute the command
        result = os.system(command)
        if result == 0:
            print(f"Successfully completed: {description}\n")
            return True
        else:
            print(f"Error executing command for: {description}\n")
            return False
    except Exception as e:
        print(f"Exception during execution: {e}")
        return False

def focus_stack_images_strategy1(output_dir: Path, strategy_suffix: str = "all") -> Path:
    """
    Strategy 1: Stack all images in folder (original implementation)
    
    Args:
        output_dir (Path): Path to the main output directory
        strategy_suffix (str): Suffix to append to output filenames
        
    Returns:
        Path: Path to the directory containing focus stacked images
    """
    # Create focus_stacked directory if it doesn't exist
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist")
    
    # Stack all images in each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith("focus_stacked"):
            folder_name = folder.name
            output_file = focus_stacked_dir / f"{folder_name}_{strategy_suffix}.tiff"
            
            # Get all image files in the folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']:
                image_files.extend(folder.glob(ext))
                image_files.extend(folder.glob(ext.upper()))
            
            if not image_files:
                print(f"No image files found in {folder_name}")
                continue
            
            description = f"Strategy 1 - {folder_name} (all {len(image_files)} images)"
            execute_focus_stack(image_files, output_file, description)
    
    return focus_stacked_dir

def stack_selected_images(output_dir: Path, folder_name: str, selected_images: List[Tuple[Path, float]], 
                         strategy_suffix: str, description: str) -> bool:
    """
    Stack a selected set of images by passing them directly to focus-stack.exe.
    
    Args:
        output_dir (Path): Main output directory
        folder_name (str): Name of the source folder
        selected_images (List[Tuple[Path, float]]): List of (image_path, f_score) tuples
        strategy_suffix (str): Suffix for output filename
        description (str): Description for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if len(selected_images) < 2:
        print(f"Warning: Only {len(selected_images)} suitable images found in {folder_name}, skipping stacking")
        return False
    
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Extract just the file paths from the (path, score) tuples
    image_files = [img_path for img_path, _ in selected_images]
    output_file = focus_stacked_dir / f"{folder_name}_{strategy_suffix}.tiff"
    
    return execute_focus_stack(image_files, output_file, description)

def focus_stack_images_strategy2(output_dir: Path, n: int = 5, strategy_suffix: str = "top_n") -> Path:
    """
    Strategy 2: Stack top N images in folder based on F score
    
    Args:
        output_dir (Path): Path to the main output directory
        n (int): Number of top images to stack
        strategy_suffix (str): Suffix to append to output filenames
        
    Returns:
        Path: Path to the directory containing focus stacked images
    """
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist")
    
    # Process each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith("focus_stacked"):
            folder_name = folder.name
            
            # Get images with F scores
            images_with_scores = get_images_with_f_scores(folder)
            
            if not images_with_scores:
                print(f"No images with F scores found in {folder_name}")
                continue
            
            # Sort by F score (descending) and take top N
            images_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_n_images = images_with_scores[:n]
            
            print(f"Strategy 2 - Processing folder: {folder_name}")
            print(f"Found {len(images_with_scores)} images with F scores")
            print(f"Using top {len(top_n_images)} images:")
            for img_path, f_score in top_n_images:
                print(f"  - {img_path.name} (F={f_score})")
            
            description = f"Strategy 2 - {folder_name} with top {len(top_n_images)} images"
            stack_selected_images(output_dir, folder_name, top_n_images, strategy_suffix, description)
    
    return focus_stacked_dir

def focus_stack_images_threshold(output_dir: Path, threshold: float = None, 
                                threshold_type: str = "absolute", strategy_suffix: str = "threshold") -> Path:
    """
    Strategy 3: Stack images above a threshold F score
    
    Args:
        output_dir (Path): Path to the main output directory
        threshold (float): F score threshold. If None, calculated automatically
        threshold_type (str): "absolute" for fixed threshold, "relative" for percentage of max
        strategy_suffix (str): Suffix to append to output filenames
        
    Returns:
        Path: Path to the directory containing focus stacked images
    """
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist")
    
    # Process each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith("focus_stacked"):
            folder_name = folder.name
            
            # Get images with F scores
            images_with_scores = get_images_with_f_scores(folder)
            
            if not images_with_scores:
                print(f"No images with F scores found in {folder_name}")
                continue
            
            # Calculate threshold if not provided
            f_scores = [score for _, score in images_with_scores]
            
            if threshold is None:
                if threshold_type == "relative":
                    # Use 70% of maximum F score as threshold
                    calculated_threshold = max(f_scores) * 0.7
                else:
                    # Use mean + 0.5 * std dev as threshold
                    mean_score = statistics.mean(f_scores)
                    std_score = statistics.stdev(f_scores) if len(f_scores) > 1 else 0
                    calculated_threshold = mean_score + (0.5 * std_score)
            else:
                if threshold_type == "relative":
                    calculated_threshold = max(f_scores) * threshold
                else:
                    calculated_threshold = threshold
            
            # Filter images above threshold
            suitable_images = [(path, score) for path, score in images_with_scores 
                             if score >= calculated_threshold]
            
            print(f"Strategy 3 - Processing folder: {folder_name}")
            print(f"Found {len(images_with_scores)} images with F scores")
            print(f"F score range: {min(f_scores):.2f} - {max(f_scores):.2f}")
            print(f"Calculated threshold: {calculated_threshold:.2f}")
            print(f"Images above threshold: {len(suitable_images)}")
            
            for img_path, f_score in suitable_images:
                print(f"  - {img_path.name} (F={f_score:.2f})")
            
            description = f"Strategy 3 - {folder_name} with {len(suitable_images)} suitable images (threshold: {calculated_threshold:.2f})"
            stack_selected_images(output_dir, folder_name, suitable_images, strategy_suffix, description)
    
    return focus_stacked_dir

# Keep all your existing stitching functions unchanged
def stitch_images(focus_stacked_dir: Path, output_dir: Path) -> None:
    """
    Stitch together focus stacked images into a panorama.
    
    Args:
        focus_stacked_dir (Path): Directory containing the focus stacked images
        output_dir (Path): Directory where the final stitched image will be saved
    """
    print("Starting stitching process...")
    
    # Get all PNG files in the focus_stacked directory
    stacked_images = list(focus_stacked_dir.glob("*.png"))
    if not stacked_images:
        raise ValueError("No stacked images found to stitch!")

    # Convert Path objects to strings
    stacked_images = [str(img) for img in stacked_images]
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2,
        "blender_type": "no",
        "warper_type": "affine",
    }    

    try:
        # Create and run the stitcher
        stitcher = AffineStitcher(**settings)
        panorama = stitcher.stitch(stacked_images)
        
        # Save the final stitched image in the output directory
        output_path = output_dir / "final_stitched.tiff"
        cv.imwrite(str(output_path), panorama)
        print(f"Successfully created stitched image: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error during stitching: {e}")

def process_folders_with_all_strategies(n: int = 5, threshold: float = None, 
                                      threshold_type: str = "absolute"):
    """
    Main function to process folders with ALL focus stacking strategies.
    
    Args:
        n (int): Number of top images for strategy 2
        threshold (float): Threshold value for strategy 3
        threshold_type (str): "absolute" or "relative" for threshold strategy
    """
    # Get the absolute path to the output directory
    output_dir = Path('output').absolute()
    
    print("=" * 80)
    print("PROCESSING WITH ALL FOCUS STACKING STRATEGIES")
    print("=" * 80)
    
    strategies_results = {}
    
    try:
        # Strategy 1: Stack all images
        print("\n" + "=" * 40)
        print("STRATEGY 1: ALL IMAGES")
        print("=" * 40)
        strategies_results['all'] = focus_stack_images_strategy1(output_dir, "all")
        
        # Strategy 2: Stack top N images by F score
        print("\n" + "=" * 40)
        print(f"STRATEGY 2: TOP {n} IMAGES BY F-SCORE")
        print("=" * 40)
        strategies_results['top_n'] = focus_stack_images_strategy2(output_dir, n, f"top{n}")
        
        # Strategy 3: Threshold-based (automatic threshold)
        print("\n" + "=" * 40)
        print("STRATEGY 3: THRESHOLD-BASED (AUTO)")
        print("=" * 40)
        strategies_results['threshold_auto'] = focus_stack_images_threshold(
            output_dir, None, "absolute", "thresh_auto"
        )
        
        # Strategy 3 variant: Relative threshold (70% of max)
        print("\n" + "=" * 40)
        print("STRATEGY 3: THRESHOLD-BASED (70% OF MAX)")  
        print("=" * 40)
        strategies_results['threshold_relative'] = focus_stack_images_threshold(
            output_dir, 0.7, "relative", "thresh_70pct"
        )
        
        # If custom threshold provided, run that too
        if threshold is not None:
            print("\n" + "=" * 40)
            print(f"STRATEGY 3: THRESHOLD-BASED (CUSTOM: {threshold})")
            print("=" * 40)
            strategies_results['threshold_custom'] = focus_stack_images_threshold(
                output_dir, threshold, threshold_type, f"thresh_{threshold}"
            )
        
        print("\n" + "=" * 80)
        print("ALL STRATEGIES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print summary
        focus_stacked_dir = output_dir / "focus_stacked"
        if focus_stacked_dir.exists():
            stacked_files = list(focus_stacked_dir.glob("*.png"))
            print(f"\nGenerated {len(stacked_files)} stacked images:")
            for file in sorted(stacked_files):
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

def process_folders_with_strategy(strategy: str = "all", n: int = 5, 
                                threshold: float = None, threshold_type: str = "absolute"):
    """
    Main function to process folders with a specific focus stacking strategy.
    
    Args:
        strategy (str): "all", "top_n", or "threshold"
        n (int): Number of top images for strategy 2
        threshold (float): Threshold value for strategy 3
        threshold_type (str): "absolute" or "relative" for threshold strategy
    """
    # Get the absolute path to the output directory
    output_dir = Path('output').absolute()
    
    try:
        if strategy == "all":
            focus_stacked_dir = focus_stack_images_strategy1(output_dir, "all")
        elif strategy == "top_n":
            focus_stacked_dir = focus_stack_images_strategy2(output_dir, n, f"top{n}")
        elif strategy == "threshold":
            focus_stacked_dir = focus_stack_images_threshold(output_dir, threshold, threshold_type, "threshold")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Focus stacking completed using strategy: {strategy}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

def stitch_focus_stacked_images(output_dir: Path = None, strategy: str = "all", 
                               stitch_method: str = "panorama"):
    """
    Stitch together focus-stacked images from a specific strategy.
    
    Args:
        output_dir (Path): Path to the output directory. If None, uses 'output'
        strategy (str): Which strategy's images to stitch ("all", "top5", "thresh_auto", 
                       "thresh_70pct", or "all_strategies" to stitch all available)
        stitch_method (str): "panorama" for full panorama, "by_y" for row-wise, "by_x" for column-wise
    """
    if output_dir is None:
        output_dir = Path('output').absolute()
    
    focus_stacked_dir = output_dir / "focus_stacked"
    
    if not focus_stacked_dir.exists():
        print(f"Focus stacked directory not found: {focus_stacked_dir}")
        print("Please run focus stacking first.")
        return
    
    print(f"Stitching focus-stacked images using strategy: {strategy}")
    print(f"Stitch method: {stitch_method}")
    
    # Get images based on strategy
    if strategy == "all_strategies":
        # Stitch all available focus-stacked images regardless of strategy
        stacked_images = list(focus_stacked_dir.glob("*.png"))
        output_suffix = "all_strategies"
    else:
        # Stitch only images from specific strategy
        stacked_images = list(focus_stacked_dir.glob(f"*_{strategy}.tiff"))
        output_suffix = strategy
    
    if not stacked_images:
        print(f"No focus-stacked images found for strategy: {strategy}")
        available_files = list(focus_stacked_dir.glob("*.png"))
        if available_files:
            print("Available files:")
            for file in available_files:
                print(f"  - {file.name}")
        return
    
    print(f"Found {len(stacked_images)} images to stitch:")
    for img in sorted(stacked_images):
        print(f"  - {img.name}")
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2,
        "blender_type": "no",
        "warper_type": "affine",
    }
    
    try:
        if stitch_method == "panorama":
            # Stitch all images into one panorama
            stacked_images_str = [str(img) for img in stacked_images]
            stitcher = AffineStitcher(**settings)
            panorama = stitcher.stitch(stacked_images_str)
            
            output_path = output_dir / f"final_stitched_{output_suffix}.tiff"
            cv.imwrite(str(output_path), panorama)
            print(f"Successfully created panorama: {output_path}")
            
        elif stitch_method == "by_y":
            # Stitch images that share the same y-position (row-wise)
            stitch_by_y_position_from_stacked(focus_stacked_dir, output_dir, strategy)
            
        elif stitch_method == "by_x":
            # Stitch images that share the same x-position (column-wise)
            stitch_by_x_position_from_stacked(focus_stacked_dir, output_dir, strategy)
            
        else:
            raise ValueError(f"Unknown stitch method: {stitch_method}")
            
    except Exception as e:
        print(f"Error during stitching: {e}")

def stitch_by_y_position_from_stacked(focus_stacked_dir: Path, output_dir: Path, strategy: str) -> None:
    """
    Stitch focus-stacked images by y-position (row-wise stitching).
    
    Args:
        focus_stacked_dir (Path): Directory containing focus-stacked images
        output_dir (Path): Output directory for stitched results
        strategy (str): Strategy suffix to filter images
    """
    # Create output directory
    y_stitched_dir = output_dir / f"y_stitched_{strategy}"
    y_stitched_dir.mkdir(exist_ok=True)
    
    # Get images for this strategy
    if strategy == "all_strategies":
        stacked_images = list(focus_stacked_dir.glob("*.tiff"))
    else:
        stacked_images = list(focus_stacked_dir.glob(f"*_{strategy}.tiff"))
    
    # Group images by y-position
    images_by_y = {}
    for image_path in stacked_images:
        try:
            # Extract folder name from focus-stacked image name
            # Format: "X{x}Y{y}_{strategy}.png" -> extract Y value
            base_name = image_path.stem  # Remove .png
            folder_name = base_name.replace(f"_{strategy}", "")  # Remove strategy suffix
            
            # Extract Y position
            y_pos = int(folder_name.split('Y')[1].split('_')[0])  # Get Y value before any underscore
            
            if y_pos not in images_by_y:
                images_by_y[y_pos] = []
            images_by_y[y_pos].append(str(image_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {image_path.name}: Could not parse coordinates - {e}")
            continue
    
    # Configure stitcher
    settings = {"crop": False, "detector": "sift", "confidence_threshold": 0.2,
        "blender_type": "no",
        "warper_type": "affine",}
    
    # Stitch each y-position group
    for y_pos, image_paths in images_by_y.items():
        if len(image_paths) < 2:
            print(f"Skipping y-position {y_pos}: Only {len(image_paths)} image(s)")
            continue
        
        try:
            # Sort by x-position
            image_paths.sort(key=lambda x: int(Path(x).stem.split('Y')[0].split('X')[1]))
            
            stitcher = AffineStitcher(**settings)
            panorama = stitcher.stitch(image_paths)
            
            output_path = y_stitched_dir / f"y_position_{y_pos}_{strategy}.tiff"
            cv.imwrite(str(output_path), panorama)
            print(f"Created y-stitched image: {output_path.name}")
            
        except Exception as e:
            print(f"Error stitching y-position {y_pos}: {e}")

def stitch_by_x_position_from_stacked(focus_stacked_dir: Path, output_dir: Path, strategy: str) -> None:
    """
    Stitch focus-stacked images by x-position (column-wise stitching).
    
    Args:
        focus_stacked_dir (Path): Directory containing focus-stacked images
        output_dir (Path): Output directory for stitched results
        strategy (str): Strategy suffix to filter images
    """
    # Create output directory
    x_stitched_dir = output_dir / f"x_stitched_{strategy}"
    x_stitched_dir.mkdir(exist_ok=True)
    
    # Get images for this strategy
    if strategy == "all_strategies":
        stacked_images = list(focus_stacked_dir.glob("*.tiff"))
    else:
        stacked_images = list(focus_stacked_dir.glob(f"*_{strategy}.tiff"))
    
    # Group images by x-position
    images_by_x = {}
    for image_path in stacked_images:
        try:
            # Extract folder name from focus-stacked image name
            base_name = image_path.stem  # Remove .png
            folder_name = base_name.replace(f"_{strategy}", "")  # Remove strategy suffix
            
            # Extract X position
            x_pos = int(folder_name.split('Y')[0].split('X')[1])  # Get X value
            
            if x_pos not in images_by_x:
                images_by_x[x_pos] = []
            images_by_x[x_pos].append(str(image_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {image_path.name}: Could not parse coordinates - {e}")
            continue
    
    # Configure stitcher
    settings = {
        "crop": False, 
        "detector": "sift", 
        "blender_type": "no",
        "warper_type": "affine",
    }
    
    # Stitch each x-position group
    for x_pos, image_paths in images_by_x.items():
        if len(image_paths) < 2:
            print(f"Skipping x-position {x_pos}: Only {len(image_paths)} image(s)")
            continue
        
        try:
            # Sort by y-position
            image_paths.sort(key=lambda x: int(Path(x).stem.split('Y')[1].split('_')[0]))
            
            stitcher = AffineStitcher(**settings)
            panorama = stitcher.stitch(image_paths)
            
            output_path = x_stitched_dir / f"x_position_{x_pos}_{strategy}.tiff"
            cv.imwrite(str(output_path), panorama)
            print(f"Created x-stitched image: {output_path.name}")
            
        except Exception as e:
            print(f"Error stitching x-position {x_pos}: {e}")

def hierarchical_stitch_by_strategy(output_dir: Path = None, strategy: str = "all"):
    """
    Hierarchically stitch focus-stacked images by pairing neighbors based on X-coordinate.
    This approach is much faster than trying to stitch all images at once.
    
    Args:
        output_dir (Path): Path to the output directory. If None, uses 'output'
        strategy (str): Which strategy's images to stitch ("all", "top5", "thresh_auto", etc.)
    """
    if output_dir is None:
        output_dir = Path('output').absolute()
    
    focus_stacked_dir = output_dir / "focus_stacked"
    
    if not focus_stacked_dir.exists():
        print(f"Focus stacked directory not found: {focus_stacked_dir}")
        print("Please run focus stacking first.")
        return
    
    # Get images for this strategy
    if strategy == "all_strategies":
        stacked_images = list(focus_stacked_dir.glob("*.tiff"))
        output_suffix = "all_strategies"
    else:
        stacked_images = list(focus_stacked_dir.glob(f"*_{strategy}.tiff"))
        output_suffix = strategy
    
    if not stacked_images:
        print(f"No focus-stacked images found for strategy: {strategy}")
        return
    
    print(f"Starting hierarchical stitching for strategy: {strategy}")
    print(f"Found {len(stacked_images)} images to stitch")
    
    # Extract X coordinates and sort images
    images_with_coords = []
    for img_path in stacked_images:
        try:
            # Extract folder name from focus-stacked image name
            base_name = img_path.stem  # Remove .png
            if strategy != "all_strategies":
                folder_name = base_name.replace(f"_{strategy}", "")  # Remove strategy suffix
            else:
                # For all_strategies, we need to extract the original folder name
                # This is trickier since different strategies have different suffixes
                folder_name = base_name
                for suffix in ["_all", "_top5", "_thresh_auto", "_thresh_70pct"]:
                    if folder_name.endswith(suffix):
                        folder_name = folder_name.replace(suffix, "")
                        break
            
            # Extract X coordinate
            x_pos = int(folder_name.split('Y')[0].split('X')[1])
            images_with_coords.append((x_pos, img_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {img_path.name}: Could not parse X coordinate - {e}")
            continue
    
    if len(images_with_coords) < 2:
        print(f"Need at least 2 images to stitch, found {len(images_with_coords)}")
        return
    
    # Sort by X coordinate
    images_with_coords.sort(key=lambda x: x[0])
    
    print("Images sorted by X coordinate:")
    for x_pos, img_path in images_with_coords:
        print(f"  X{x_pos}: {img_path.name}")
    
    # Create working directory for intermediate results
    working_dir = output_dir / f"stitch_working_{output_suffix}"
    working_dir.mkdir(exist_ok=True)
    
    try:
        # Start hierarchical stitching
        current_images = [img_path for _, img_path in images_with_coords]
        level = 0
        
        while len(current_images) > 1:
            level += 1
            print(f"\n--- Stitching Level {level} ---")
            print(f"Processing {len(current_images)} images")
            
            next_level_images = []
            pairs_processed = 0
            
            # Process pairs of neighboring images
            for i in range(0, len(current_images), 2):
                if i + 1 < len(current_images):
                    # Pair of images
                    img1 = current_images[i]
                    img2 = current_images[i + 1]
                    
                    # Create output filename for this pair
                    output_name = f"level{level}_pair{pairs_processed + 1}_{output_suffix}.tiff"
                    output_path = working_dir / output_name
                    
                    print(f"  Stitching pair {pairs_processed + 1}: {img1.name} + {img2.name}")
                    
                    if stitch_image_pair(img1, img2, output_path):
                        next_level_images.append(output_path)
                        pairs_processed += 1
                    else:
                        print(f"    Failed to stitch pair, skipping")
                        # If stitching fails, keep the first image for next level
                        next_level_images.append(img1)
                        
                else:
                    # Odd image out, carry to next level
                    print(f"  Carrying forward: {current_images[i].name}")
                    next_level_images.append(current_images[i])
            
            current_images = next_level_images
            print(f"Level {level} completed. {len(current_images)} images remain.")
        
        # Move final result to main output directory
        if current_images:
            final_image = current_images[0]
            final_output = output_dir / f"final_hierarchical_stitched_{output_suffix}.tiff"
            
            if final_image.parent == working_dir:
                # It's an intermediate result, move it
                final_image.rename(final_output)
            else:
                # It's an original image (only one image was provided), copy it
                import shutil
                shutil.copy2(final_image, final_output)
            
            print(f"\n🎉 Hierarchical stitching completed!")
            print(f"Final result: {final_output}")
        
    finally:
        # Clean up working directory
        import shutil
        try:
            shutil.rmtree(working_dir)
            print(f"Cleaned up working directory: {working_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up working directory: {e}")

def stitch_image_pair(img1_path: Path, img2_path: Path, output_path: Path) -> bool:
    """
    Stitch two images together.
    
    Args:
        img1_path (Path): First image
        img2_path (Path): Second image  
        output_path (Path): Output path for stitched result
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Configure stitcher for pairs (more lenient settings)
        settings = {
            "crop": False,
            "detector": "sift",
            "confidence_threshold": 0.1,  # Lower threshold for pairs
            "blender_type": "no",
        "warper_type": "affine",
        }
        
        stitcher = AffineStitcher(**settings)
        image_paths = [str(img1_path), str(img2_path)]
        
        panorama = stitcher.stitch(image_paths)
        cv.imwrite(str(output_path), panorama)
        
        return True
        
    except Exception as e:
        print(f"    Error stitching {img1_path.name} + {img2_path.name}: {e}")
        return False

def stitch_all_strategies_hierarchically(output_dir: Path = None):
    """
    Run hierarchical stitching for all available strategies.
    
    Args:
        output_dir (Path): Path to the output directory. If None, uses 'output'
    """
    if output_dir is None:
        output_dir = Path('output').absolute()
    
    focus_stacked_dir = output_dir / "focus_stacked"
    
    if not focus_stacked_dir.exists():
        print("No focus-stacked images found. Please run focus stacking first.")
        return
    
    # Find available strategies
    all_files = list(focus_stacked_dir.glob("*.png"))
    strategies = set()
    
    for file in all_files:
        name = file.stem
        for suffix in ["_all", "_top5", "_thresh_auto", "_thresh_70pct"]:
            if name.endswith(suffix):
                strategies.add(suffix[1:])  # Remove leading underscore
                break
    
    if not strategies:
        print("No recognizable strategy suffixes found in focus-stacked images")
        return
    
    print(f"Found strategies: {', '.join(sorted(strategies))}")
    
    # Stitch each strategy
    for strategy in sorted(strategies):
        print(f"\n{'=' * 60}")
        print(f"HIERARCHICAL STITCHING - STRATEGY: {strategy.upper()}")
        print(f"{'=' * 60}")
        hierarchical_stitch_by_strategy(output_dir, strategy)

def select_best_single_images(output_dir: Path, strategy_suffix: str = "best_single") -> Path:
    """
    Strategy 4: Select only the single best image (highest F-score) from each folder.
    No focus stacking - just picks the sharpest individual image.
    
    Args:
        output_dir (Path): Path to the main output directory
        strategy_suffix (str): Suffix to append to output filenames
        
    Returns:
        Path: Path to the directory containing selected best images
    """
    # Create best_images directory if it doesn't exist
    best_images_dir = output_dir / "best_images"
    best_images_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist")
    
    selected_images = []
    
    # Process each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith(("focus_stacked", "best_images", "stitch_working", "y_stitched", "x_stitched")):
            folder_name = folder.name
            
            # Get images with F scores
            images_with_scores = get_images_with_f_scores(folder)
            
            if not images_with_scores:
                print(f"No images with F scores found in {folder_name}")
                continue
            
            # Find the image with the highest F score
            best_image_path, best_f_score = max(images_with_scores, key=lambda x: x[1])
            
            print(f"Strategy 4 - Best image from {folder_name}:")
            print(f"  Selected: {best_image_path.name} (F={best_f_score:.2f})")
            print(f"  Out of {len(images_with_scores)} images (F-score range: {min(score for _, score in images_with_scores):.2f} - {max(score for _, score in images_with_scores):.2f})")
            
            # Copy the best image to the best_images directory with folder name
            output_file = best_images_dir / f"{folder_name}_{strategy_suffix}.tiff"
            
            try:
                import shutil
                shutil.copy2(best_image_path, output_file)
                selected_images.append((folder_name, output_file, best_f_score))
                print(f"  Copied to: {output_file.name}\n")
                
            except Exception as e:
                print(f"Error copying {best_image_path} to {output_file}: {e}")
                continue
    
    print(f"Selected {len(selected_images)} best images:")
    for folder_name, img_path, f_score in selected_images:
        print(f"  {folder_name}: F={f_score:.2f}")
    
    return best_images_dir

def hierarchical_stitch_best_images(output_dir: Path = None, strategy: str = "best_single"):
    """
    Hierarchically stitch the best single images (no focus stacking involved).
    This should produce the sharpest results since we're using the original sharpest images.
    
    Args:
        output_dir (Path): Path to the output directory. If None, uses 'output'
        strategy (str): Strategy suffix for the images to stitch
    """
    if output_dir is None:
        output_dir = Path('output').absolute()
    
    best_images_dir = output_dir / "best_images"
    
    if not best_images_dir.exists():
        print(f"Best images directory not found: {best_images_dir}")
        print("Please run select_best_single_images() first.")
        return
    
    # Get images for this strategy
    best_images = list(best_images_dir.glob(f"*_{strategy}.tiff"))
    
    if not best_images:
        print(f"No best images found for strategy: {strategy}")
        return
    
    print(f"Starting hierarchical stitching of best single images")
    print(f"Found {len(best_images)} images to stitch")
    
    # Extract X coordinates and sort images
    images_with_coords = []
    for img_path in best_images:
        try:
            # Extract folder name from best image name
            base_name = img_path.stem  # Remove .png
            folder_name = base_name.replace(f"_{strategy}", "")  # Remove strategy suffix
            
            # Extract X coordinate
            x_pos = int(folder_name.split('Y')[0].split('X')[1])
            images_with_coords.append((x_pos, img_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {img_path.name}: Could not parse X coordinate - {e}")
            continue
    
    if len(images_with_coords) < 2:
        print(f"Need at least 2 images to stitch, found {len(images_with_coords)}")
        return
    
    # Sort by X coordinate
    images_with_coords.sort(key=lambda x: x[0])
    
    print("Best images sorted by X coordinate:")
    for x_pos, img_path in images_with_coords:
        print(f"  X{x_pos}: {img_path.name}")
    
    # Create working directory for intermediate results
    working_dir = output_dir / f"stitch_working_{strategy}"
    working_dir.mkdir(exist_ok=True)
    
    try:
        # Start hierarchical stitching
        current_images = [img_path for _, img_path in images_with_coords]
        level = 0
        
        while len(current_images) > 1:
            level += 1
            print(f"\n--- Stitching Level {level} ---")
            print(f"Processing {len(current_images)} images")
            
            next_level_images = []
            pairs_processed = 0
            
            # Process pairs of neighboring images
            for i in range(0, len(current_images), 2):
                if i + 1 < len(current_images):
                    # Pair of images
                    img1 = current_images[i]
                    img2 = current_images[i + 1]
                    
                    # Create output filename for this pair
                    output_name = f"level{level}_pair{pairs_processed + 1}_{strategy}.tiff"
                    output_path = working_dir / output_name
                    
                    print(f"  Stitching pair {pairs_processed + 1}: {img1.name} + {img2.name}")
                    
                    if stitch_image_pair(img1, img2, output_path):
                        next_level_images.append(output_path)
                        pairs_processed += 1
                    else:
                        print(f"    Failed to stitch pair, skipping")
                        # If stitching fails, keep the first image for next level
                        next_level_images.append(img1)
                        
                else:
                    # Odd image out, carry to next level
                    print(f"  Carrying forward: {current_images[i].name}")
                    next_level_images.append(current_images[i])
            
            current_images = next_level_images
            print(f"Level {level} completed. {len(current_images)} images remain.")
        
        # Move final result to main output directory
        if current_images:
            final_image = current_images[0]
            final_output = output_dir / f"final_best_single_stitched.tiff"
            
            if final_image.parent == working_dir:
                # It's an intermediate result, move it
                final_image.rename(final_output)
            else:
                # It's an original image (only one image was provided), copy it
                import shutil
                shutil.copy2(final_image, final_output)
            
            print(f"\n🎉 Best single image stitching completed!")
            print(f"Final result: {final_output}")
            print(f"This should be the sharpest result since it uses original unprocessed images!")
        
    finally:
        # Clean up working directory
        import shutil
        try:
            shutil.rmtree(working_dir)
            print(f"Cleaned up working directory: {working_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up working directory: {e}")

def process_folders_with_all_strategies_plus_best(n: int = 5, threshold: float = None, 
                                                 threshold_type: str = "absolute"):
    """
    Main function to process folders with ALL focus stacking strategies PLUS best single image strategy.
    
    Args:
        n (int): Number of top images for strategy 2
        threshold (float): Threshold value for strategy 3
        threshold_type (str): "absolute" or "relative" for threshold strategy
    """
    # Get the absolute path to the output directory
    output_dir = Path('output').absolute()
    
    print("=" * 80)
    print("PROCESSING WITH ALL STRATEGIES + BEST SINGLE IMAGE")
    print("=" * 80)
    
    try:
        # Strategy 1: Stack all images
        print("\n" + "=" * 40)
        print("STRATEGY 1: ALL IMAGES")
        print("=" * 40)
        focus_stack_images_strategy1(output_dir, "all")
        
        # Strategy 2: Stack top N images by F score
        print("\n" + "=" * 40)
        print(f"STRATEGY 2: TOP {n} IMAGES BY F-SCORE")
        print("=" * 40)
        focus_stack_images_strategy2(output_dir, n, f"top{n}")
        
        # Strategy 3: Threshold-based (automatic threshold)
        print("\n" + "=" * 40)
        print("STRATEGY 3: THRESHOLD-BASED (AUTO)")
        print("=" * 40)
        focus_stack_images_threshold(output_dir, None, "absolute", "thresh_auto")
        
        # Strategy 3 variant: Relative threshold (70% of max)
        print("\n" + "=" * 40)
        print("STRATEGY 3: THRESHOLD-BASED (70% OF MAX)")  
        print("=" * 40)
        focus_stack_images_threshold(output_dir, 0.7, "relative", "thresh_70pct")
        
        # Strategy 4: Best single image (NEW!)
        print("\n" + "=" * 40)
        print("STRATEGY 4: BEST SINGLE IMAGE (NO STACKING)")
        print("=" * 40)
        select_best_single_images(output_dir, "best_single")
        
        # If custom threshold provided, run that too
        if threshold is not None:
            print("\n" + "=" * 40)
            print(f"STRATEGY 3: THRESHOLD-BASED (CUSTOM: {threshold})")
            print("=" * 40)
            focus_stack_images_threshold(output_dir, threshold, threshold_type, f"thresh_{threshold}")
        
        print("\n" + "=" * 80)
        print("ALL STRATEGIES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print summary
        focus_stacked_dir = output_dir / "focus_stacked"
        best_images_dir = output_dir / "best_images"
        
        if focus_stacked_dir.exists():
            stacked_files = list(focus_stacked_dir.glob("*.png"))
            print(f"\nGenerated {len(stacked_files)} focus-stacked images:")
            for file in sorted(stacked_files):
                print(f"  - {file.name}")
        
        if best_images_dir.exists():
            best_files = list(best_images_dir.glob("*.png"))
            print(f"\nSelected {len(best_files)} best single images:")
            for file in sorted(best_files):
                print(f"  - {file.name}")
        
        # Now run hierarchical stitching for best single images
        print("\n" + "=" * 80)
        print("HIERARCHICAL STITCHING - BEST SINGLE IMAGES")
        print("=" * 80)
        hierarchical_stitch_best_images(output_dir, "best_single")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    # Run only the best single image strategy and stitch
    output_dir = Path('output').absolute()
    
    print("=" * 80)
    print("BEST SINGLE IMAGE STRATEGY + STITCHING")
    print("=" * 80)
    
    try:
        # Strategy 4: Best single image
        print("\n" + "=" * 40)
        print("STRATEGY 4: BEST SINGLE IMAGE (NO STACKING)")
        print("=" * 40)
        select_best_single_images(output_dir, "best_single")
        
        # Hierarchical stitching of best single images
        print("\n" + "=" * 40)
        print("HIERARCHICAL STITCHING - BEST SINGLE IMAGES")
        print("=" * 40)
        hierarchical_stitch_best_images(output_dir, "best_single")
        
        print("\n" + "=" * 80)
        print("BEST SINGLE IMAGE PROCESSING COMPLETED!")
        print("=" * 80)
        
        # Print summary
        best_images_dir = output_dir / "best_images"
        if best_images_dir.exists():
            best_files = list(best_images_dir.glob("*.png"))
            print(f"\nSelected {len(best_files)} best single images:")
            for file in sorted(best_files):
                print(f"  - {file.name}")
        
        final_result = output_dir / "final_best_single_stitched.png"
        if final_result.exists():
            print(f"\nFinal stitched result: {final_result}")
            print("This should be your sharpest panorama!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    
    # Comment out other usage examples:
    # You can also run other strategies individually if needed:
    # process_folders_with_all_strategies_plus_best(n=5)
    # select_best_single_images(Path('output').absolute())
    # hierarchical_stitch_best_images(strategy="best_single")