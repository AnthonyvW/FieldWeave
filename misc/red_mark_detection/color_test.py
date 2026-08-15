import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.signal import find_peaks

def detect_red_marks(image_path, visualize=True):
    """
    Detect red registration marks in an image with shiny black background.
    
    Args:
        image_path: Path to the input image
        visualize: Whether to show visualization plots
    
    Returns:
        centers: List of (x, y) coordinates of detected mark centers
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract channels
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    
    # Enhanced red isolation: R - max(G, B) to handle reflections better
    red_isolated = r - np.maximum(g, b)
    red_isolated = np.clip(red_isolated, 0, 255)
    
    # Normalize to 0-255 range
    if red_isolated.max() > 0:
        red_isolated = (red_isolated / red_isolated.max() * 255).astype(np.uint8)
    else:
        red_isolated = red_isolated.astype(np.uint8)
    
    # Apply morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_isolated = cv2.morphologyEx(red_isolated, cv2.MORPH_CLOSE, kernel)
    
    # Adaptive thresholding to handle varying lighting
    threshold = np.percentile(red_isolated[red_isolated > 0], 85) if np.any(red_isolated > 0) else 50
    binary = (red_isolated > threshold).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter components by size and aspect ratio
    centers = []
    min_area = 50  # Adjust based on expected mark size
    max_area = 10000
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            
            # Registration marks are typically reasonably circular/square
            if aspect_ratio < 3.0:
                centers.append((centroids[i][0], centroids[i][1]))
    
    if visualize:
        visualize_detection(img_rgb, red_isolated, binary, centers)
    
    return centers

def visualize_detection(img_rgb, red_isolated, binary, centers):
    """Visualize the detection process and results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Red isolated
    axes[0, 1].imshow(red_isolated, cmap='hot')
    axes[0, 1].set_title('Red Channel Isolation')
    axes[0, 1].axis('off')
    
    # Binary threshold
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary Threshold')
    axes[0, 2].axis('off')
    
    # X-axis projection (sum along columns)
    x_projection = np.sum(red_isolated, axis=0)
    axes[1, 0].plot(x_projection)
    axes[1, 0].set_title('X-axis Projection')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Intensity Sum')
    
    # Y-axis projection (sum along rows)
    y_projection = np.sum(red_isolated, axis=1)
    axes[1, 1].plot(y_projection)
    axes[1, 1].set_title('Y-axis Projection')
    axes[1, 1].set_xlabel('Y Position')
    axes[1, 1].set_ylabel('Intensity Sum')
    
    # Result with detected centers
    axes[1, 2].imshow(img_rgb)
    
    # Draw individual center dots
    for x, y in centers:
        axes[1, 2].plot(x, y, 'ro', markersize=5)
    
    # Calculate and draw the mean center line
    if centers:
        mean_x = np.mean([x for x, y in centers])
        axes[1, 2].axvline(x=mean_x, color='yellow', linewidth=2, linestyle='--', label=f'Center: x={mean_x:.2f}')
        axes[1, 2].legend()
    
    axes[1, 2].set_title(f'Detected Centers ({len(centers)} marks)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_folder(folder_path, save_results=True):
    """
    Process all images in a folder and detect registration marks.
    
    Args:
        folder_path: Path to folder containing images
        save_results: Whether to save results to a text file
    
    Returns:
        results: Dictionary mapping image names to detected centers
    """
    folder = Path(folder_path)
    results = {}
    
    # Get all image files once
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(ext))
        image_files.extend(folder.glob(ext.upper()))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Process all images first
    all_data = []
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load and process image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract channels
        r = img_rgb[:, :, 0].astype(float)
        g = img_rgb[:, :, 1].astype(float)
        b = img_rgb[:, :, 2].astype(float)
        
        # Enhanced red isolation
        red_isolated = r - np.maximum(g, b)
        red_isolated = np.clip(red_isolated, 0, 255)
        
        if red_isolated.max() > 0:
            red_isolated = (red_isolated / red_isolated.max() * 255).astype(np.uint8)
        else:
            red_isolated = red_isolated.astype(np.uint8)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_isolated = cv2.morphologyEx(red_isolated, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive thresholding
        threshold = np.percentile(red_isolated[red_isolated > 0], 75) if np.any(red_isolated > 0) else 50
        binary = (red_isolated > threshold).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Get image dimensions
        img_height = img_rgb.shape[0]
        
        # Filter components - restrict to lower half of image
        centers = []
        filtered_centers = []  # Store filtered out centers
        min_area = 50
        max_area = 10000
        lower_half_y = img_height / 2
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                
                if aspect_ratio < 3.0:
                    center = (centroids[i][0], centroids[i][1])
                    # Check if in lower half
                    if center[1] >= lower_half_y:
                        centers.append(center)
                    else:
                        filtered_centers.append(center)
        
        # Remove outliers using IQR method on X coordinates
        valid_centers = centers.copy()
        outlier_centers = []
        
        if len(centers) > 3:  # Need at least 4 points for meaningful outlier detection
            x_coords = np.array([x for x, y in centers])
            q1 = np.percentile(x_coords, 25)
            q3 = np.percentile(x_coords, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_centers = []
            outlier_centers = []
            for center in centers:
                if lower_bound <= center[0] <= upper_bound:
                    valid_centers.append(center)
                else:
                    outlier_centers.append(center)
        
        # Combine all filtered centers
        all_filtered = filtered_centers + outlier_centers
        
        results[img_path.name] = valid_centers
        all_data.append({
            'name': img_path.name,
            'img_rgb': img_rgb,
            'red_isolated': red_isolated,
            'binary': binary,
            'centers': valid_centers,
            'filtered_centers': all_filtered
        })
        
        print(f"  Detected {len(valid_centers)} valid marks ({len(all_filtered)} filtered)")
        if valid_centers:
            mean_x = np.mean([x for x, y in valid_centers])
            print(f"  Mean center X: {mean_x:.2f}")
    
    # Create interactive viewer
    if all_data:
        create_interactive_viewer(all_data)
    
    if save_results:
        output_file = folder / "detection_results.txt"
        with open(output_file, 'w') as f:
            for img_name, centers in results.items():
                f.write(f"{img_name}:\n")
                for i, (x, y) in enumerate(centers):
                    f.write(f"  Mark {i+1}: ({x:.2f}, {y:.2f})\n")
                if centers:
                    mean_x = np.mean([x for x, y in centers])
                    f.write(f"  Mean center X: {mean_x:.2f}\n")
                f.write("\n")
        print(f"\nResults saved to: {output_file}")
    
    return results

def create_interactive_viewer(all_data):
    """Create an interactive viewer to cycle through images."""
    current_idx = [0]  # Use list to allow modification in nested function
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    def update_display(idx):
        """Update the display with the current image."""
        data = all_data[idx]
        
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
        
        # Original image
        axes[0, 0].imshow(data['img_rgb'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Red isolated
        axes[0, 1].imshow(data['red_isolated'], cmap='hot')
        axes[0, 1].set_title('Red Channel Isolation')
        axes[0, 1].axis('off')
        
        # Binary threshold
        axes[0, 2].imshow(data['binary'], cmap='gray')
        axes[0, 2].set_title('Binary Threshold')
        axes[0, 2].axis('off')
        
        # X-axis projection
        x_projection = np.sum(data['red_isolated'], axis=0)
        axes[1, 0].plot(x_projection)
        
        # Add percentile lines for X-axis
        if np.any(x_projection > 0):
            p25 = np.percentile(x_projection, 25)
            p50 = np.percentile(x_projection, 50)
            p75 = np.percentile(x_projection, 75)
            axes[1, 0].axhline(y=p25, color='cyan', linestyle=':', linewidth=1, label='25th')
            axes[1, 0].axhline(y=p50, color='orange', linestyle=':', linewidth=1, label='50th')
            axes[1, 0].axhline(y=p75, color='magenta', linestyle=':', linewidth=1, label='75th')
            axes[1, 0].legend(loc='upper right', fontsize=8)
        
        axes[1, 0].set_title('X-axis Projection')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Intensity Sum')
        
        # Y-axis projection
        y_projection = np.sum(data['red_isolated'], axis=1)
        axes[1, 1].plot(y_projection)
        
        # Add percentile lines for Y-axis
        if np.any(y_projection > 0):
            p25 = np.percentile(y_projection, 25)
            p50 = np.percentile(y_projection, 50)
            p75 = np.percentile(y_projection, 75)
            axes[1, 1].axhline(y=p25, color='cyan', linestyle=':', linewidth=1, label='25th')
            axes[1, 1].axhline(y=p50, color='orange', linestyle=':', linewidth=1, label='50th')
            axes[1, 1].axhline(y=p75, color='magenta', linestyle=':', linewidth=1, label='75th')
            axes[1, 1].legend(loc='upper right', fontsize=8)
        
        axes[1, 1].set_title('Y-axis Projection')
        axes[1, 1].set_xlabel('Y Position')
        axes[1, 1].set_ylabel('Intensity Sum')
        
        # Result with detected centers
        axes[1, 2].imshow(data['img_rgb'])
        
        # Draw filtered out centers in red
        for x, y in data['filtered_centers']:
            axes[1, 2].plot(x, y, 'ro', markersize=5, alpha=0.6)
        
        # Draw valid center dots in bright green
        for x, y in data['centers']:
            axes[1, 2].plot(x, y, 'o', color='lime', markersize=5)
        
        # Calculate and draw the mean center line
        if data['centers']:
            mean_x = np.mean([x for x, y in data['centers']])
            axes[1, 2].axvline(x=mean_x, color='yellow', linewidth=2, linestyle='--', label=f'Center: x={mean_x:.2f}')
            axes[1, 2].legend()
        
        title_text = f'Valid: {len(data["centers"])} marks'
        if data['filtered_centers']:
            title_text += f' | Filtered: {len(data["filtered_centers"])}'
        axes[1, 2].set_title(title_text)
        axes[1, 2].axis('off')
        
        fig.suptitle(f'{data["name"]} ({idx + 1}/{len(all_data)})', fontsize=14, fontweight='bold')
        fig.canvas.draw()
    
    def on_key(event):
        """Handle keyboard navigation."""
        if event.key == 'right' or event.key == 'down':
            current_idx[0] = (current_idx[0] + 1) % len(all_data)
            update_display(current_idx[0])
        elif event.key == 'left' or event.key == 'up':
            current_idx[0] = (current_idx[0] - 1) % len(all_data)
            update_display(current_idx[0])
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(0)
    plt.tight_layout()
    print("\nUse arrow keys (left/right or up/down) to navigate between images")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Process all images in the folder
    results = process_folder("./test_img/", save_results=True)
    
    # Or process a single image
    # centers = detect_red_marks("./test_img/your_image.jpg", visualize=True)