import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional

def load_texture(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a texture image and optionally resize it."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size:
        img = cv2.resize(img, size)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Texture')
    plt.show()
    return np.array(img) / 255.0

def visualize_progress(synthesized, mask, step=0):
    """Visualize the original texture and synthesis progress."""
    plt.figure(figsize=(5, 2))
    plt.subplot(121)
    plt.imshow(synthesized, cmap='gray')
    plt.title('Synthesized Texture')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title("Filled Pixel")
    plt.axis('off')
    plt.suptitle(f"Step {step}", y=0)
    plt.show()

def get_unfilled_neighbors_count(mask: np.ndarray, window_size: int) -> np.ndarray:
    """Count number of filled neighbors for each unfilled pixel."""
    kernel = np.ones((window_size, window_size))
    neighbors_count = cv2.filter2D(mask.astype(float), -1, kernel)
    neighbors_count[mask > 0] = -1  # Mark filled pixels with -1
    return neighbors_count

def create_gaussian_window(window_size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a Gaussian window for neighborhood weighting."""
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    half_size = window_size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    xx, yy = np.meshgrid(x, y)
    
    # Gaussian formula: exp(-(x² + y²)/(2σ²))
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize so max value is 1.0
    gaussian /= gaussian.max()
    
    return gaussian

def find_matching_neighborhoods(target_neighborhood: np.ndarray,
                               source_texture: np.ndarray,
                               gaussian_window: np.ndarray,
                               valid_mask: np.ndarray,
                               threshold: float = 0.01) -> list:
    """Find matching neighborhoods in source texture."""
    window_size = target_neighborhood.shape[0]
    half_size = window_size // 2
    h, w = source_texture.shape
    
    good_matches = []
    
    # Calculate valid weights sum for normalization
    valid_weights = gaussian_window * valid_mask
    total_valid_weight = np.sum(valid_weights)
    
    if total_valid_weight == 0:
        return good_matches
    
    # Slide window over source texture
    for i in range(half_size, h - half_size):
        for j in range(half_size, w - half_size):
            # Extract source neighborhood
            source_neighborhood = source_texture[i-half_size:i+half_size+1, 
                                               j-half_size:j+half_size+1]
            
            # Calculate weighted SSD (Sum of Squared Differences)
            diff = (target_neighborhood - source_neighborhood) * valid_mask
            weighted_diff = diff**2 * gaussian_window
            ssd = np.sum(weighted_diff) / total_valid_weight
            
            # Check if match is good enough
            if ssd <= threshold:
                good_matches.append((i, j))
    
    return good_matches

def synthesize_texture(source_texture: np.ndarray,
                      output_size: Tuple[int, int],
                      window_size: int = 15,
                      threshold: float = 0.01,
                      visualize: bool = True) -> np.ndarray:
    """Synthesize a new texture using Efros & Leung's algorithm."""
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half_size = window_size // 2
    output_h, output_w = output_size
    source_h, source_w = source_texture.shape
    
    # Initialize output and mask
    output = np.zeros(output_size)
    mask = np.zeros(output_size, dtype=bool)
    
    # Create Gaussian window
    gaussian_window = create_gaussian_window(window_size)
    
    # Place seed in center
    center_h, center_w = output_h // 2, output_w // 2
    seed_size = 3
    
    # Copy random 3x3 patch from source as seed
    seed_i = np.random.randint(half_size, source_h - half_size - seed_size)
    seed_j = np.random.randint(half_size, source_w - half_size - seed_size)
    
    seed_patch = source_texture[seed_i:seed_i+seed_size, seed_j:seed_j+seed_size]
    output[center_h:center_h+seed_size, center_w:center_w+seed_size] = seed_patch
    mask[center_h:center_h+seed_size, center_w:center_w+seed_size] = True
    
    # Calculate total number of pixels to fill
    total_pixels = output_h * output_w
    filled_pixels = seed_size * seed_size
    
    # Main synthesis loop
    iteration = 0
    while filled_pixels < total_pixels:
        # Find pixels with most filled neighbors
        neighbors_count = get_unfilled_neighbors_count(mask, window_size)
        
        if np.max(neighbors_count) == -1:  # All pixels filled
            break
            
        # Find pixel with maximum filled neighbors
        max_count = np.max(neighbors_count)
        candidate_indices = np.where(neighbors_count == max_count)
        
        if len(candidate_indices[0]) == 0:
            break
            
        # Choose random candidate
        idx = np.random.randint(len(candidate_indices[0]))
        i, j = candidate_indices[0][idx], candidate_indices[1][idx]
        
        # Extract target neighborhood
        i_min = max(0, i - half_size)
        i_max = min(output_h, i + half_size + 1)
        j_min = max(0, j - half_size)
        j_max = min(output_w, j + half_size + 1)
        
        target_neighborhood = np.zeros((window_size, window_size))
        valid_mask = np.zeros((window_size, window_size), dtype=bool)
        
        # Fill target neighborhood and valid mask
        start_i = half_size - (i - i_min)
        start_j = half_size - (j - j_min)
        end_i = start_i + (i_max - i_min)
        end_j = start_j + (j_max - j_min)
        
        target_neighborhood[start_i:end_i, start_j:end_j] = output[i_min:i_max, j_min:j_max]
        valid_mask[start_i:end_i, start_j:end_j] = mask[i_min:i_max, j_min:j_max]
        
        # Find matching neighborhoods
        matches = find_matching_neighborhoods(target_neighborhood, source_texture, 
                                            gaussian_window, valid_mask, threshold)
        
        if matches:
            # Randomly select a match
            match_i, match_j = matches[np.random.randint(len(matches))]
            output[i, j] = source_texture[match_i, match_j]
            mask[i, j] = True
            filled_pixels += 1
        
        # Visualize progress
        if iteration % 1000 == 0 and visualize:
            visualize_progress(output, mask, step=iteration)
        
        iteration += 1
    
    # if visualize:
    #     visualize_progress(output, mask, step=iteration)
    
    return output

# Test the Gaussian window
def test_gaussian_window():
    window = create_gaussian_window(7, sigma=1.0)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(window, cmap='viridis')
    plt.colorbar()
    plt.title('Gaussian Window')
    plt.subplot(122)
    plt.plot(window[window.shape[0]//2, :])
    plt.title('Center Cross Section')
    plt.grid(True)
    plt.show()
    
    print("Window shape:", window.shape)
    print("Max value (should be 1.0):", window.max())
    print("Center value (should be 1.0):", window[window.shape[0]//2, window.shape[0]//2])
    print("Corner value (should be small):", window[0,0])
    print("Window is symmetric:", np.allclose(window, window.T))

# Test neighborhood matching
def test_neighborhood_matching():
    # Create a simple test texture
    source = np.zeros((20, 20))
    source[5:15, 5:15] = 1  # White square in center
    
    # Create a target neighborhood that should match the edge of the square
    window_size = 5
    target = np.zeros((window_size, window_size))
    target[:, :3] = 0
    target[:, 3:] = 1
    
    # Create valid mask (all pixels valid in this test)
    valid_mask = np.ones_like(target)
    
    # Create Gaussian window
    gaussian = create_gaussian_window(window_size)
    
    # Find matches
    matches = find_matching_neighborhoods(target, source, gaussian, valid_mask)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(source, cmap='gray')
    plt.title('Source Texture')
    
    # Highlight matches
    for i, j in matches:
        rect = plt.Rectangle((j-window_size//2, i-window_size//2),
                            window_size, window_size,
                            fill=False, color='red')
        plt.gca().add_patch(rect)
    
    plt.subplot(132)
    plt.imshow(target, cmap='gray')
    plt.title('Target Neighborhood')
    plt.subplot(133)
    plt.imshow(valid_mask * gaussian, cmap='gray')
    plt.title('Valid Mask × Gaussian')
    plt.show()
    
    print(f"Found {len(matches)} matches")

# Run tests
print("Testing Gaussian Window:")
test_gaussian_window()

print("\nTesting Neighborhood Matching:")
test_neighborhood_matching()

# Test with actual texture synthesis
print("\nTesting Texture Synthesis:")
# Create a simple test pattern
test_texture = np.zeros((32, 32))
test_texture[8:24, 8:24] = 1
test_texture[12:20, 12:20] = 0.5

synthesized = synthesize_texture(test_texture, (64, 64), window_size=9, 
                               threshold=0.05, visualize=True)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(test_texture, cmap='gray')
plt.title('Original Texture')
plt.subplot(122)
plt.imshow(synthesized, cmap='gray')
plt.title('Synthesized Texture')
plt.show()