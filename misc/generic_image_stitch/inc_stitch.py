#!/usr/bin/env python3
"""
Progressive Image Stitching Viewer
Shows images being stitched together in real-time using pygame.
Improved version with template matching fallback for vertical alignment.
"""

import cv2
import numpy as np
import os
import argparse
import pygame
import time
from pathlib import Path


class ProgressiveStitcher:
    def __init__(self, display_width=1200, display_height=800, row_counts=None):
        """Initialize the progressive stitcher with pygame display."""
        pygame.init()
        self.display_width = display_width
        self.display_height = display_height
        self.screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Progressive Image Stitching")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Spatial layout information
        self.row_counts = row_counts or []  # Number of images per row
        
        # World space for images
        self.images = []  # List of dicts with 'image', 'world_pos', 'index'
        self.world_offset = [0, 0]  # Camera offset for panning
        self.zoom_scale = 1.0
        
        # Track current position in snake pattern
        self.current_row = 0
        self.current_col = 0
        self.direction = -1  # -1 for left, 1 for right
        
        # Estimated image size (will be updated with first image)
        self.avg_image_width = 800
        self.avg_image_height = 600
        self.overlap_ratio = 0.4  # Estimated overlap between images
        
    def load_images_from_folder(self, folder_path):
        """Load all JPEG images from the specified folder in order."""
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        
        # Get all image files
        all_files = []
        for file in os.listdir(folder_path):
            if any(file.endswith(ext) for ext in valid_extensions):
                all_files.append(file)
        
        # Sort by numeric prefix if possible
        def get_numeric_key(filename):
            # Extract leading numbers from filename
            import re
            match = re.match(r'(\d+)', filename)
            if match:
                return int(match.group(1))
            return filename
        
        all_files.sort(key=get_numeric_key)
        
        image_files = [os.path.join(folder_path, f) for f in all_files]
        
        if not image_files:
            raise ValueError(f"No JPEG images found in {folder_path}")
        
        print(f"Found {len(image_files)} images:")
        for img_file in image_files:
            print(f"  - {os.path.basename(img_file)}")
        
        return image_files
    
    def find_feature_rich_region(self, image_gray, direction):
        """
        Use edge detection to find the most feature-rich vertical strip in the image.
        
        Args:
            image_gray: Grayscale image
            direction: -1 for left, 1 for right (which side to prioritize)
            
        Returns:
            tuple: (start_col, width) for the best region, or None if detection fails
        """
        h, w = image_gray.shape
        
        # Detect edges using Canny
        edges = cv2.Canny(image_gray, 50, 150)
        
        # Divide image into vertical strips and count edges in each
        num_strips = 10
        strip_width = w // num_strips
        edge_counts = []
        
        for i in range(num_strips):
            start_x = i * strip_width
            end_x = min((i + 1) * strip_width, w)
            strip = edges[:, start_x:end_x]
            edge_count = np.sum(strip > 0)
            edge_counts.append((i, edge_count, start_x, end_x))
        
        # Sort by edge count (most features first)
        edge_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Prioritize strips on the side we're interested in
        if direction == -1:
            # Going left: prefer left side (lower indices)
            # Weight: lower index = higher priority
            weighted_scores = [(i, count - (idx * count * 0.1), start_x, end_x) 
                              for idx, count, start_x, end_x in edge_counts]
        else:
            # Going right: prefer right side (higher indices)
            # Weight: higher index = higher priority
            weighted_scores = [(i, count + (idx * count * 0.1), start_x, end_x) 
                              for idx, count, start_x, end_x in edge_counts]
        
        # Re-sort by weighted score
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3-4 strips and use them (they might be adjacent)
        best_strips = weighted_scores[:4]
        best_indices = [x[0] for x in best_strips]
        best_indices.sort()
        
        # Find contiguous region
        if len(best_indices) >= 2:
            start_idx = best_indices[0]
            end_idx = best_indices[-1]
            # Expand to include strips in between
            start_col = start_idx * strip_width
            end_col = min((end_idx + 1) * strip_width, w)
            width = end_col - start_col
            
            # Ensure width is reasonable (20-50% of image)
            min_width = int(w * 0.2)
            max_width = int(w * 0.5)
            if width < min_width:
                width = min_width
            if width > max_width:
                width = max_width
                # Adjust start_col if needed
                if direction == -1:
                    start_col = 0
                else:
                    start_col = w - width
            
            print(f"    Edge detection: Best region at x={start_col}-{start_col + width} ({width}px wide)")
            return start_col, width
        
        return None

    def find_vertical_offset_template_matching(self, new_image, prev_row_image):
        """
        Use template matching to find vertical offset between images.
        Also detects horizontal offset based on where the match occurred.
        
        Args:
            new_image: The new image (top)
            prev_row_image: The image from previous row (bottom)
            
        Returns:
            tuple: (y_offset, x_offset, confidence) or (None, None, 0) if matching fails
        """
        print(f"    Attempting template matching for vertical alignment...")
        
        new_h, new_w = new_image.shape[:2]
        prev_h, prev_w = prev_row_image.shape[:2]
        
        # Convert to grayscale for template matching
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_row_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate estimated overlap based on overlap ratio
        estimated_overlap_height = int(self.avg_image_height * self.overlap_ratio)
        
        # Use a more conservative template height (60% of estimated overlap)
        # This ensures the template is significantly smaller than the search region
        overlap_height = int(estimated_overlap_height * 0.6)
        overlap_height = min(overlap_height, int(new_h * 0.3))  # Cap at 30% of image height
        template_full = new_gray[-overlap_height:, :]
        
        # Search region should be generous: estimated overlap + 100% margin
        search_margin = estimated_overlap_height  # Full overlap height as margin
        search_height = estimated_overlap_height + search_margin
        search_height = min(search_height, prev_h)  # Don't exceed image bounds
        
        # Use edge detection to find feature-rich region for the template
        feature_region = self.find_feature_rich_region(template_full, self.direction)
        
        if feature_region is not None:
            feature_start_col, feature_width = feature_region
            
            # Crop template to feature-rich region
            template = template_full[:, feature_start_col:feature_start_col + feature_width]
            
            print(f"    Using edge-detected template region: {feature_width}px wide at x={feature_start_col}")
            
            # Store where we cropped the template from (for offset calculation later)
            template_crop_start = feature_start_col
            template_crop_width = feature_width
        else:
            # Fallback: use fixed percentage on the side based on direction
            print(f"    Edge detection failed, using fallback template crop")
            template_crop_ratio = 0.3
            template_crop_width = int(new_w * template_crop_ratio)
            
            if self.direction == -1:
                # Going left: use left side
                template = template_full[:, :template_crop_width]
                template_crop_start = 0
            else:
                # Going right: use right side
                template = template_full[:, -template_crop_width:]
                template_crop_start = new_w - template_crop_width
        
        # Search region: use the same side as template, but wider
        # This ensures the template region will be found in the search region
        search_crop_ratio = 0.5  # Use 50% of the width for search
        search_crop_width = int(prev_w * search_crop_ratio)
        
        if self.direction == -1:
            # Going left: search in left portion
            search_region_full = prev_gray[:search_height, :]
            search_region = search_region_full[:, :search_crop_width]
            search_crop_start = 0
            print(f"    Direction: LEFT, searching in left {search_crop_ratio:.0%} ({search_crop_width}px)")
        else:
            # Going right: search in right portion
            search_region_full = prev_gray[:search_height, :]
            search_region = search_region_full[:, -search_crop_width:]
            search_crop_start = prev_w - search_crop_width
            print(f"    Direction: RIGHT, searching in right {search_crop_ratio:.0%} ({search_crop_width}px)")
        
        print(f"    Template: {overlap_height}x{template_crop_width}px, Search: {search_height}x{search_crop_width}px")
        
        print(f"    Using estimated overlap: {estimated_overlap_height}px, template: {overlap_height}px height")
        
        # Resize if images are too large for efficient matching
        # Scale based on the TEMPLATE size to keep it manageable
        max_dim = 500
        scale = 1.0
        template_h_before_scale, template_w_before_scale = template.shape
        template_max_dim = max(template_h_before_scale, template_w_before_scale)
        if template_max_dim > max_dim:
            scale = max_dim / template_max_dim
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            search_region = cv2.resize(search_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        template_h, template_w = template.shape
        search_h, search_w = search_region.shape
        
        print(f"    After scaling: Template {template_h}x{template_w}, Search {search_h}x{search_w}")
        
        # Save debug images to see what's being matched
        debug_dir = "template_matching_debug"
        import os
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save template and search region
        cv2.imwrite(f"{debug_dir}/template_{self.current_row}.png", template)
        cv2.imwrite(f"{debug_dir}/search_{self.current_row}.png", search_region)
        print(f"    Debug: Saved template and search images to {debug_dir}/")
        
        # OpenCV requires search region to be strictly larger than template in BOTH dimensions
        if template_h >= search_h or template_w >= search_w:
            print(f"    Template matching failed: template too large")
            return None, None, 0
        
        # Additional check: ensure there's meaningful room for searching (template < 70% of search)
        if template_h > search_h * 0.7 or template_w > search_w * 0.7:
            print(f"    Template matching failed: insufficient search space (template needs to be <70% of search)")
            return None, None, 0
        
        # Perform template matching using normalized cross-correlation
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"    Template matching confidence: {max_val:.3f}")
        
        # If confidence is too low and we used edge detection, try fallback with side-based crop
        if max_val < 0.5 and feature_region is not None:
            print(f"    Edge-detected template confidence too low, trying side-based fallback...")
            
            # Try side-based template instead
            fallback_crop_ratio = 0.4
            fallback_crop_width = int(new_w * fallback_crop_ratio)
            
            if self.direction == -1:
                template_fallback = template_full[:, :fallback_crop_width]
                template_crop_start = 0
            else:
                template_fallback = template_full[:, -fallback_crop_width:]
                template_crop_start = new_w - fallback_crop_width
            
            # Resize fallback template
            template_h_fb, template_w_fb = template_fallback.shape
            template_max_dim_fb = max(template_h_fb, template_w_fb)
            scale_fb = 1.0
            if template_max_dim_fb > max_dim:
                scale_fb = max_dim / template_max_dim_fb
                template_fallback = cv2.resize(template_fallback, None, fx=scale_fb, fy=scale_fb, interpolation=cv2.INTER_AREA)
                search_region_fb = cv2.resize(search_region, None, fx=scale_fb, fy=scale_fb, interpolation=cv2.INTER_AREA)
            else:
                search_region_fb = search_region.copy()
            
            template_h_fb, template_w_fb = template_fallback.shape
            search_h_fb, search_w_fb = search_region_fb.shape
            
            # Check if valid
            if template_h_fb < search_h_fb and template_w_fb < search_w_fb and \
               template_h_fb < search_h_fb * 0.7 and template_w_fb < search_w_fb * 0.7:
                
                result_fb = cv2.matchTemplate(search_region_fb, template_fallback, cv2.TM_CCOEFF_NORMED)
                min_val_fb, max_val_fb, min_loc_fb, max_loc_fb = cv2.minMaxLoc(result_fb)
                
                print(f"    Fallback confidence: {max_val_fb:.3f}")
                
                # Use fallback if it's better
                if max_val_fb > max_val:
                    print(f"    Using fallback template (better confidence)")
                    template = template_fallback
                    search_region = search_region_fb
                    max_val = max_val_fb
                    max_loc = max_loc_fb
                    scale = scale_fb
                    template_h, template_w = template_h_fb, template_w_fb
                    search_h, search_w = search_h_fb, search_w_fb
                    
                    # Save fallback debug images
                    cv2.imwrite(f"{debug_dir}/template_{self.current_row}_fallback.png", template)
        
        # Require a minimum confidence threshold
        if max_val < 0.5:
            print(f"    Template matching confidence too low")
            return None, None, 0
        
        # The match location tells us where the TOP-LEFT of template matched in the search region
        # Template is the bottom overlap_height pixels of new_image
        # Search region is the top search_height pixels of prev_row_image
        match_y = max_loc[1]
        match_x = max_loc[0]
        
        # Scale back to original resolution
        match_y = int(match_y / scale)
        match_x = int(match_x / scale)
        
        # Calculate the Y offset:
        # The template (bottom of new image) matched at position match_y in the search region
        # Y offset = match_y + overlap_height - new_h (should be negative)
        y_offset = match_y + overlap_height - new_h
        
        # Calculate the X offset:
        # match_x tells us where the template matched within the search region (in search_region coordinates)
        # We need to convert this to world coordinates accounting for our crops
        # Template was cropped starting at template_crop_start
        # Search was cropped starting at search_crop_start
        # If match_x = 0, it means template aligned perfectly with search start
        # The actual X offset in world coordinates:
        x_offset = search_crop_start + match_x - template_crop_start
        
        # Create visualization showing where the match was found
        debug_dir = "template_matching_debug"
        
        # Also save edge detection visualization
        edges_new = cv2.Canny(new_gray, 50, 150)
        edges_prev = cv2.Canny(prev_gray, 50, 150)
        cv2.imwrite(f"{debug_dir}/edges_new_{self.current_row}.png", edges_new)
        cv2.imwrite(f"{debug_dir}/edges_prev_{self.current_row}.png", edges_prev)
        
        vis_search = cv2.cvtColor(search_region, cv2.COLOR_GRAY2BGR)
        # Draw rectangle where template matched
        cv2.rectangle(vis_search, 
                     (match_x, match_y), 
                     (match_x + template_w, match_y + template_h),
                     (0, 255, 0), 2)
        # Add text showing match position
        cv2.putText(vis_search, f"Match: ({match_x}, {match_y})", 
                   (match_x, match_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f"{debug_dir}/match_visualization_{self.current_row}.png", vis_search)
        
        print(f"    Template match found at x={match_x}, y={match_y}")
        print(f"    Template height={overlap_height}, new image height={new_h}")
        print(f"    Calculated offsets: X={x_offset:.1f}, Y={y_offset:.1f} pixels")
        
        return y_offset, x_offset, max_val
    
    def find_horizontal_offset_template_matching(self, new_image, prev_image, direction):
        """
        Use template matching to find horizontal offset between images.
        
        Args:
            new_image: The new image
            prev_image: The previous image
            direction: -1 for left, 1 for right
            
        Returns:
            tuple: (x_offset, confidence) or (None, 0) if matching fails
        """
        print(f"    Attempting template matching for horizontal alignment...")
        
        new_h, new_w = new_image.shape[:2]
        prev_h, prev_w = prev_image.shape[:2]
        
        # Convert to grayscale
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate estimated overlap based on overlap ratio
        estimated_overlap_width = int(self.avg_image_width * self.overlap_ratio)
        
        # Take overlap region based on direction
        # Use estimated overlap width, but cap it at 40% of image width for safety
        overlap_width = min(estimated_overlap_width, int(new_w * 0.4))
        
        # Focus search region around expected overlap area
        search_margin = int(estimated_overlap_width * 0.5)
        search_width = estimated_overlap_width + search_margin
        search_width = min(search_width, prev_w)  # Don't exceed image bounds
        
        if direction == -1:
            # Moving left: take right edge of new image as template
            template = new_gray[:, -overlap_width:]
            # Search in left portion of previous image (focused on overlap region)
            search_region = prev_gray[:, :search_width]
        else:
            # Moving right: take left edge of new image as template
            template = new_gray[:, :overlap_width]
            # Search in right portion of previous image (focused on overlap region)
            search_region = prev_gray[:, -search_width:]
        
        print(f"    Using estimated overlap: {estimated_overlap_width}px, template: {overlap_width}px, search: {search_width}px")
        
        # Resize if images are too large
        max_height = 600
        scale = 1.0
        if new_h > max_height or prev_h > max_height:
            scale = max_height / max(new_h, prev_h)
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            search_region = cv2.resize(search_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        template_h, template_w = template.shape
        search_h, search_w = search_region.shape
        
        # OpenCV requires search region to be strictly larger than template in BOTH dimensions
        if template_w >= search_w or template_h >= search_h:
            print(f"    Template matching failed: template too large (template: {template_h}x{template_w}, search: {search_h}x{search_w})")
            return None, 0
        
        # Additional check: ensure there's meaningful room for searching
        if template_w > search_w * 0.9 or template_h > search_h * 0.95:
            print(f"    Template matching failed: insufficient search space (template: {template_h}x{template_w}, search: {search_h}x{search_w})")
            return None, 0
        
        # Perform template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"    Template matching confidence: {max_val:.3f}")
        
        if max_val < 0.5:
            print(f"    Template matching confidence too low")
            return None, 0
        
        match_x = max_loc[0]
        match_x = int(match_x / scale)
        
        # Calculate x offset
        if direction == -1:
            # Moving left: new image goes to the left of previous
            x_offset = -(prev_w - match_x)
        else:
            # Moving right: new image goes to the right of previous
            x_offset = (prev_w - search_width) + match_x
        
        print(f"    Template match found at x={match_x}, offset={x_offset:.1f} pixels")
        
        return x_offset, max_val
    
    def get_nearby_images(self, world_pos, radius=2.5):
        """Get images within radius of the given world position."""
        nearby = []
        px, py = world_pos
        
        for img_data in self.images:
            ix, iy = img_data['world_pos']
            # Calculate distance in "image units"
            dist_x = abs(px - ix) / self.avg_image_width
            dist_y = abs(py - iy) / self.avg_image_height
            dist = (dist_x**2 + dist_y**2)**0.5
            
            if dist <= radius:
                nearby.append(img_data)
        
        return nearby
    
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates."""
        wx, wy = world_pos
        sx = (wx + self.world_offset[0]) * self.zoom_scale + self.display_width // 2
        sy = (wy + self.world_offset[1]) * self.zoom_scale + self.display_height // 2
        return [sx, sy]
    
    def render_worldspace(self):
        """Render all images in their worldspace positions."""
        self.screen.fill((30, 30, 30))
        
        # Sort images by index to render in order (earlier images first)
        sorted_images = sorted(self.images, key=lambda x: x['index'])
        
        for img_data in sorted_images:
            image = img_data['image']
            world_pos = img_data['world_pos']
            
            # Convert to screen space
            screen_pos = self.world_to_screen(world_pos)
            
            # Scale image
            h, w = image.shape[:2]
            scaled_w = int(w * self.zoom_scale)
            scaled_h = int(h * self.zoom_scale)
            
            # Skip if too small or off screen
            if scaled_w < 2 or scaled_h < 2:
                continue
            if screen_pos[0] + scaled_w < 0 or screen_pos[0] > self.display_width:
                continue
            if screen_pos[1] + scaled_h < 0 or screen_pos[1] > self.display_height:
                continue
            
            # Resize and convert to pygame
            scaled_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
            # Transpose to get correct orientation for pygame (swap axes)
            rgb_image = np.transpose(rgb_image, (1, 0, 2))
            surface = pygame.surfarray.make_surface(rgb_image)
            
            # Draw with slight transparency for overlaps
            surface.set_alpha(220)
            self.screen.blit(surface, screen_pos)
            
            # Draw border
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (*screen_pos, scaled_w, scaled_h), 1)
    
    def auto_frame_images(self):
        """Automatically adjust zoom and offset to fit all images."""
        if not self.images:
            return
        
        # Find bounding box of all images
        min_x = min(img['world_pos'][0] for img in self.images)
        max_x = max(img['world_pos'][0] + img['image'].shape[1] for img in self.images)
        min_y = min(img['world_pos'][1] for img in self.images)
        max_y = max(img['world_pos'][1] + img['image'].shape[0] for img in self.images)
        
        # Calculate zoom to fit
        width = max_x - min_x
        height = max_y - min_y
        
        zoom_x = (self.display_width - 100) / width if width > 0 else 1.0
        zoom_y = (self.display_height - 150) / height if height > 0 else 1.0
        self.zoom_scale = min(zoom_x, zoom_y, 1.0)
        
        # Center on images
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.world_offset = [-center_x, -center_y]
    
    def draw_status_overlay(self, message, image_num, total_images, loading_file=None):
        """Draw status text overlay on top of rendered scene."""
        # Draw semi-transparent background for text
        overlay = pygame.Surface((self.display_width, 130), pygame.SRCALPHA)
        overlay.fill((30, 30, 30, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Draw main message
        text = self.font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.display_width // 2, 30))
        self.screen.blit(text, text_rect)
        
        # Draw progress
        progress_text = f"Image {image_num} of {total_images}"
        progress = self.small_font.render(progress_text, True, (200, 200, 200))
        progress_rect = progress.get_rect(center=(self.display_width // 2, 70))
        self.screen.blit(progress, progress_rect)
        
        # Draw loading indicator if provided
        if loading_file:
            loading_text = f"Loading: {loading_file}"
            loading = self.small_font.render(loading_text, True, (150, 200, 255))
            loading_rect = loading.get_rect(center=(self.display_width // 2, 100))
            self.screen.blit(loading, loading_rect)
    
    def add_image(self, new_image, index):
        """Add a new image to the worldspace using stitching to determine precise position."""
        h, w = new_image.shape[:2]
        
        if not self.images:
            # First image at origin
            self.avg_image_width = w
            self.avg_image_height = h
            self.current_col = self.row_counts[0] - 1 if self.row_counts else 0
            self.current_row = 0
            self.direction = -1
            print(f"  First image size: {w}x{h}")
            
            self.images.append({
                'image': new_image,
                'world_pos': [0, 0],
                'index': index,
                'row': 0,
                'col': self.current_col
            })
            return True
        
        # Check if we're moving to a new row
        need_new_row = False
        if self.direction == -1:
            # Moving left
            if self.current_col == 0:
                # About to finish this row
                need_new_row = True
        else:
            # Moving right
            if self.current_row < len(self.row_counts) and self.current_col == self.row_counts[self.current_row] - 1:
                # About to finish this row
                need_new_row = True
        
        if need_new_row:
            # Moving to new row - need to stitch with image from previous row
            print(f"  Moving to new row {self.current_row + 1}")
            
            # Get the last image from the current row (this is where we transition from)
            transition_image = self.images[-1]
            
            self.current_row += 1
            self.direction = -self.direction  # Flip direction
            
            if self.direction == -1:
                # Now moving left, start at rightmost column of new row
                self.current_col = self.row_counts[self.current_row] - 1 if self.current_row < len(self.row_counts) else 0
            else:
                # Now moving right, start at leftmost column of new row
                self.current_col = 0
            
            # Find the appropriate image from the previous row to stitch with
            prev_row_images = [img for img in self.images if img.get('row', 0) == self.current_row - 1]
            
            if prev_row_images:
                transition_x = transition_image['world_pos'][0]
                
                # Find candidate images from previous row to try matching against
                # Sort by X distance from transition point
                candidates = sorted(prev_row_images, key=lambda img: abs(img['world_pos'][0] - transition_x))
                
                # Try up to 3 candidates from previous row
                max_candidates = min(3, len(candidates))
                
                print(f"  Will try matching against {max_candidates} candidates from previous row")
                
                best_match = None
                best_confidence = 0
                
                for i, candidate_img in enumerate(candidates[:max_candidates]):
                    print(f"  Candidate {i+1}/{max_candidates}: image {candidate_img['index']} (X: {candidate_img['world_pos'][0]:.1f})")
                    
                    # First try SIFT-based stitching
                    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
                    status, stitched = stitcher.stitch([new_image, candidate_img['image']])
                    
                    if status == cv2.Stitcher_OK:
                        # Calculate Y offset from stitched result
                        stitched_h, stitched_w = stitched.shape[:2]
                        prev_h, prev_w = candidate_img['image'].shape[:2]
                        
                        # Y offset is negative (moving up)
                        y_offset = -(stitched_h - prev_h)
                        next_y = candidate_img['world_pos'][1] + y_offset
                        
                        # X position: use the candidate image's X position
                        next_x = candidate_img['world_pos'][0]
                        
                        print(f"    SIFT successful! Y offset: {y_offset:.1f} pixels")
                        world_pos = [next_x, next_y]
                        break  # Found a good match, stop searching
                    else:
                        # SIFT failed, try template matching
                        print(f"    SIFT failed, trying template matching...")
                        y_offset, x_offset, confidence = self.find_vertical_offset_template_matching(
                            new_image, candidate_img['image']
                        )
                        
                        if confidence > best_confidence:
                            best_match = {
                                'candidate': candidate_img,
                                'y_offset': y_offset,
                                'x_offset': x_offset,
                                'confidence': confidence
                            }
                            best_confidence = confidence
                            print(f"    Template confidence: {confidence:.3f} (new best)")
                        else:
                            print(f"    Template confidence: {confidence:.3f}")
                        
                        # If we found a very good match, stop searching
                        if confidence > 0.8:
                            break
                
                # Check if we found a good match through SIFT (world_pos was set) or template matching
                if 'world_pos' not in locals():
                    # SIFT didn't work for any candidate, use best template match
                    if best_match and best_confidence > 0.5:
                        next_y = best_match['candidate']['world_pos'][1] + best_match['y_offset']
                        next_x = best_match['candidate']['world_pos'][0] + best_match['x_offset']
                        print(f"  Using best template match: image {best_match['candidate']['index']}, confidence: {best_confidence:.3f}")
                        print(f"  Offsets: X={best_match['x_offset']:.1f}, Y={best_match['y_offset']:.1f} pixels")
                        world_pos = [next_x, next_y]
                    else:
                        # All methods failed, use estimated offset
                        closest_img = candidates[0]
                        y_offset = -(self.avg_image_height * (1 - self.overlap_ratio))
                        x_offset = 0
                        print(f"  All matching attempts failed (best confidence: {best_confidence:.3f}), using estimated offset")
                        print(f"  Estimated offsets: X={x_offset:.1f}, Y={y_offset:.1f} pixels")
                        next_y = closest_img['world_pos'][1] + y_offset
                        next_x = closest_img['world_pos'][0]
                        world_pos = [next_x, next_y]
            else:
                # No previous row images, use estimated offset from last image
                prev_pos = self.images[-1]['world_pos']
                y_offset = -(self.avg_image_height * (1 - self.overlap_ratio))
                x_offset = 0  # No horizontal offset for new row
                print(f"  No previous row images found, using estimated offset")
                print(f"  Estimated offsets: X={x_offset:.1f}, Y={y_offset:.1f} pixels")
                world_pos = [prev_pos[0], prev_pos[1] + y_offset]
        
        else:
            # Continue in same row - stitch horizontally with previous image
            prev_image_data = self.images[-1]
            prev_image = prev_image_data['image']
            prev_pos = prev_image_data['world_pos']
            
            # Use stitcher to find homography/transformation
            stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
            
            # Stitch the two images
            if self.direction == -1:
                # Moving left: prev_image on right, new_image on left
                status, stitched = stitcher.stitch([new_image, prev_image])
            else:
                # Moving right: prev_image on left, new_image on right
                status, stitched = stitcher.stitch([prev_image, new_image])
            
            if status == cv2.Stitcher_OK:
                # Calculate offset based on stitched result
                stitched_h, stitched_w = stitched.shape[:2]
                prev_h, prev_w = prev_image.shape[:2]
                new_h, new_w = new_image.shape[:2]
                
                if self.direction == -1:
                    # Moving left: new image adds width to the left
                    x_offset = -(stitched_w - prev_w)
                    next_x = prev_pos[0] + x_offset
                    next_y = prev_pos[1]
                    print(f"  Horizontal stitch (SIFT) successful! X offset: {x_offset:.1f} pixels (moving left)")
                else:
                    # Moving right: new image adds width to the right
                    x_offset = stitched_w - prev_w
                    next_x = prev_pos[0] + x_offset
                    next_y = prev_pos[1]
                    print(f"  Horizontal stitch (SIFT) successful! X offset: {x_offset:.1f} pixels (moving right)")
                
                world_pos = [next_x, next_y]
            else:
                # SIFT failed, try template matching
                print(f"  Horizontal stitch (SIFT) failed, trying template matching...")
                x_offset, confidence = self.find_horizontal_offset_template_matching(
                    new_image, prev_image, self.direction
                )
                
                if x_offset is not None and confidence > 0.5:
                    next_x = prev_pos[0] + x_offset
                    next_y = prev_pos[1]
                    print(f"  Template matching successful! X offset: {x_offset:.1f} pixels (confidence: {confidence:.3f})")
                    world_pos = [next_x, next_y]
                else:
                    # Both methods failed, use estimated position
                    x_offset = self.avg_image_width * (1 - self.overlap_ratio) * self.direction
                    y_offset = 0  # No vertical offset in same row
                    print(f"  Template matching also failed, using estimated position")
                    print(f"  Estimated offsets: X={x_offset:.1f}, Y={y_offset:.1f} pixels")
                    next_x = prev_pos[0] + x_offset
                    next_y = prev_pos[1]
                    world_pos = [next_x, next_y]
            
            # Update column counter
            if self.direction == -1:
                self.current_col -= 1
            else:
                self.current_col += 1
        
        print(f"  Placing at world position: ({world_pos[0]:.1f}, {world_pos[1]:.1f}), row: {self.current_row}, col: {self.current_col}")
        
        # Add image to worldspace
        self.images.append({
            'image': new_image,
            'world_pos': world_pos,
            'index': index,
            'row': self.current_row,
            'col': self.current_col
        })
        
        return True
    
    def run(self, image_files, delay=1.0):
        """Run the progressive stitching visualization."""
        total_images = len(image_files)
        running = True
        
        try:
            for idx, img_file in enumerate(image_files, 1):
                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                
                if not running:
                    break
                
                # Show loading message while keeping previous view
                filename = os.path.basename(img_file)
                print(f"\nLoading image {idx}/{total_images}: {filename}")
                
                if self.images:
                    # Render current worldspace with loading indicator
                    self.render_worldspace()
                    self.draw_status_overlay(
                        f"Panorama Progress: {len(self.images)}/{total_images} images",
                        len(self.images),
                        total_images,
                        loading_file=filename
                    )
                    pygame.display.flip()
                
                # Simulate capture delay
                start_time = time.time()
                while time.time() - start_time < delay:
                    # Check for quit events during delay
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                break
                    
                    if not running:
                        break
                    
                    time.sleep(0.01)
                
                if not running:
                    break
                
                # Load image
                img = cv2.imread(img_file)
                
                if img is None:
                    print(f"Warning: Could not load {img_file}, skipping...")
                    continue
                
                # Add image to worldspace
                print(f"Adding image {idx} to worldspace...")
                self.add_image(img, idx)
                
                # Auto-frame to show all images
                self.auto_frame_images()
                
                # Render updated worldspace
                self.render_worldspace()
                self.draw_status_overlay(
                    f"Panorama Progress: {len(self.images)}/{total_images} images",
                    len(self.images),
                    total_images
                )
                pygame.display.flip()
                
                # Small pause to show the result
                time.sleep(0.3)
            
            # Final display
            if running and self.images:
                print("\nAll images loaded!")
                self.render_worldspace()
                self.draw_status_overlay(
                    "All Images Loaded! (Press ESC or close window to exit)",
                    total_images,
                    total_images
                )
                pygame.display.flip()
                
                # Wait for user to close
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                waiting = False
                    self.clock.tick(30)
        
        finally:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Progressive image stitching viewer with real-time display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inc_stitch_improved.py /path/to/images --rows 9 9 9 10 8 8
  python inc_stitch_improved.py /path/to/images --delay 2.0 --rows 9 9 9 10 8 8
  python inc_stitch_improved.py /path/to/images --width 1600 --height 900 --rows 9 9 9 10 8 8
        """
    )
    
    parser.add_argument('input_folder', type=str,
                       help='Folder containing JPEG images to stitch')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                       help='Delay between loading images in seconds (default: 1.0)')
    parser.add_argument('--width', '-w', type=int, default=1200,
                       help='Display width in pixels (default: 1200)')
    parser.add_argument('--height', '-ht', type=int, default=800,
                       help='Display height in pixels (default: 800)')
    parser.add_argument('--rows', '-r', type=int, nargs='+', default=None,
                       help='Number of images per row in snake pattern (e.g., 9 9 9 10 8 8)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return 1
    
    try:
        # Create stitcher
        stitcher = ProgressiveStitcher(args.width, args.height, row_counts=args.rows)
        
        # Load image file paths
        image_files = stitcher.load_images_from_folder(args.input_folder)
        
        if len(image_files) < 1:
            print("Error: Need at least 1 image")
            return 1
        
        # Validate row counts if provided
        if args.rows:
            total_expected = sum(args.rows)
            if total_expected != len(image_files):
                print(f"Warning: Row counts sum to {total_expected} but found {len(image_files)} images")
                print("Proceeding anyway...")
        
        # Run progressive stitching
        stitcher.run(image_files, delay=args.delay)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())