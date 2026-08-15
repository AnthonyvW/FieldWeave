import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_color_distribution(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate histograms for each color channel
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(12, 6))
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=color.upper())
    
    plt.title('Color Distribution Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate some basic statistics
    color_stats = {}
    for i, color in enumerate(colors):
        channel = img[:,:,i]
        stats = {
            'mean': np.mean(channel),
            'median': np.median(channel),
            'std': np.std(channel),
            'min': np.min(channel),
            'max': np.max(channel)
        }
        color_stats[color] = stats
    
    return format_color_stats(color_stats)

def format_color_stats(stats_dict):
    def format_number(value):
        # Handle numpy types and convert to float
        num = float(value)
        # Format with 4 decimal places if small, otherwise 2
        return f"{num:.4f}" if abs(num) < 0.01 else f"{num:.2f}"

    formatted_output = "Color Channel Statistics:\n\n"
    
    for channel, stats in stats_dict.items():
        # Convert channel name to title case and full name
        channel_name = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}[channel]
        
        formatted_output += f"{channel_name} Channel:\n"
        formatted_output += f"- Mean: {format_number(stats['mean'])} ({float(stats['mean'])*100:.2f}%)\n"
        formatted_output += f"- Median: {format_number(stats['median'])}\n"
        formatted_output += f"- Standard Deviation: {format_number(stats['std'])}\n"
        formatted_output += f"- Range: {int(stats['min'])} to {int(stats['max'])}\n\n"
    
    return formatted_output

def calculate_quadrant_focus(image, kernel_size=3, threshold=100):
    """
    Calculate focus measures for each quadrant of the image.
    
    Args:
        image: numpy array of the image
        kernel_size (int): Size of the Laplacian kernel
        threshold (float): Threshold for determining if quadrant is in focus
        
    Returns:
        tuple: (quadrant_scores, best_score, best_quadrant, histogram_data)
    """
    height, width = image.shape[:2]
    mid_h, mid_w = height // 2, width // 2
    
    # Define quadrants
    quadrants = [
        ('Top Left', image[0:mid_h, 0:mid_w]),
        ('Top Right', image[0:mid_h, mid_w:]),
        ('Bottom Left', image[mid_h:, 0:mid_w]),
        ('Bottom Right', image[mid_h:, mid_w:])
    ]
    
    quadrant_scores = {}
    all_hist_data = {}
    
    for name, quad in quadrants:
        # Convert quadrant to grayscale if it's not already
        if len(quad.shape) == 3:
            quad = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(quad, (3, 3), 0)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
        abs_laplacian = np.absolute(laplacian)
        
        # Calculate histogram
        hist, bins = np.histogram(abs_laplacian, bins=256, range=(0, 256))
        
        # Calculate focus measures
        variance = np.var(abs_laplacian)
        percentile_90 = np.percentile(abs_laplacian, 90)
        
        # Calculate focus score
        focus_score = (variance + percentile_90) / 2
        
        quadrant_scores[name] = focus_score
        all_hist_data[name] = (hist, bins)
    
    # Find best quadrant
    best_quadrant = max(quadrant_scores.items(), key=lambda x: x[1])
    
    return quadrant_scores, best_quadrant[1], best_quadrant[0], all_hist_data

def plot_focus_comparison(image_path1, image_path2):
    """
    Compare and plot focus measures for two images, analyzing quadrants separately.
    
    Args:
        image_path1 (str): Path to the first image file
        image_path2 (str): Path to the second image file
    """
    # Read images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    # Calculate focus measures for both images
    scores1, best_score1, best_quad1, hist_data1 = calculate_quadrant_focus(img1)
    scores2, best_score2, best_quad2, hist_data2 = calculate_quadrant_focus(img2)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original images
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Draw quadrant lines
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Plot images with quadrant lines
    ax1.imshow(img1_rgb)
    ax1.axhline(h1//2, color='w', alpha=0.5)
    ax1.axvline(w1//2, color='w', alpha=0.5)
    ax1.set_title(f'Image 1\nBest Quadrant: {best_quad1}\nScore: {best_score1:.2f}')
    ax1.axis('off')
    
    ax2.imshow(img2_rgb)
    ax2.axhline(h2//2, color='w', alpha=0.5)
    ax2.axvline(w2//2, color='w', alpha=0.5)
    ax2.set_title(f'Image 2\nBest Quadrant: {best_quad2}\nScore: {best_score2:.2f}')
    ax2.axis('off')
    
    # Plot histograms of best quadrants only
    hist1, bins1 = hist_data1[best_quad1]
    hist2, bins2 = hist_data2[best_quad2]
    
    # Calculate 50th percentile for both best quadrants
    percentile_50_1 = np.percentile(np.array(range(256))[hist1 > 0], 25)
    percentile_50_2 = np.percentile(np.array(range(256))[hist2 > 0], 25)
    x_max = max(percentile_50_1, percentile_50_2)
    
    # Plot best quadrant histograms
    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    
    # Calculate new bins up to 50th percentile for better visualization
    num_bins = 50
    ax3.hist(bin_centers1, bins=num_bins, weights=hist1, alpha=0.5, 
             label=f'Image 1 ({best_quad1})', range=(0, x_max), color='blue')
    ax3.hist(bin_centers2, bins=num_bins, weights=hist2, alpha=0.5, 
             label=f'Image 2 ({best_quad2})', range=(0, x_max), color='red')
    
    ax3.set_title('Best Quadrant Gradient Distribution (25th percentile limit)')
    ax3.set_xlabel('Gradient Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.set_xlim(0, x_max)
    ax3.legend()
    
    # Plot quadrant scores comparison
    quadrants = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
    scores_1 = [scores1[q] for q in quadrants]
    scores_2 = [scores2[q] for q in quadrants]
    
    x = np.arange(len(quadrants))
    width = 0.35
    
    # Create bars
    bars1 = ax4.bar(x - width/2, scores_1, width, label='Image 1', alpha=0.7)
    bars2 = ax4.bar(x + width/2, scores_2, width, label='Image 2', alpha=0.7)
    
    # Add value labels on the bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax4.set_title('Quadrant Score Comparison')
    ax4.set_ylabel('Focus Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(quadrants, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print("\nFocus Analysis Comparison (Best Quadrants):")
    print(f"\nImage 1 ({image_path1}):")
    print(f"  Best Quadrant: {best_quad1}")
    print(f"  Best Score: {best_score1:.2f}")
    print("\nQuadrant Scores:")
    for quad, score in scores1.items():
        print(f"  {quad}: {score:.2f}")
    
    print(f"\nImage 2 ({image_path2}):")
    print(f"  Best Quadrant: {best_quad2}")
    print(f"  Best Score: {best_score2:.2f}")
    print("\nQuadrant Scores:")
    for quad, score in scores2.items():
        print(f"  {quad}: {score:.2f}")
    
    # Compare best quadrants
    if best_score1 > best_score2:
        difference = ((best_score1 - best_score2) / best_score2) * 100
        print(f"\nImage 1's {best_quad1} is more in focus (by {difference:.1f}%)")
    elif best_score2 > best_score1:
        difference = ((best_score2 - best_score1) / best_score1) * 100
        print(f"\nImage 2's {best_quad2} is more in focus (by {difference:.1f}%)")
    else:
        print("\nBoth images' best quadrants have the same focus score")


# Example usage
if __name__ == "__main__":
    # Replace with your image paths
    image_path = "./output/color/"
    plot_focus_comparison(image_path + "test173PX11000Y15700Z4380.png", image_path + "test175PX11000Y15300Z4380.png")



    # # Example usage:
    # path = "./output/color/"
    # stats = analyze_color_distribution(path + 'test1PX10000Y10500Z4380.png')
    # print(stats)
    # stats = analyze_color_distribution(path + 'test55PX10200Y12900Z4380.png')
    # print(stats)