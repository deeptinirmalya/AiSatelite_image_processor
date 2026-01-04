import rasterio
import numpy as np
from rasterio.enums import Resampling
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

def load_and_preprocess(path, target_res=(3, 3), target_shape=(256, 256)):
    """
    Loads a satellite image, resamples it to target resolution, and normalizes.
    """
    with rasterio.open(path) as src:
        # Calculate scaling factor to match target resolution or shape
        count = src.count
        data = src.read(
            out_shape=(count, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear
        )
        
        # Ensure 3 channels for ResNet (C, H, W)
        if count == 1:
            # Grayscale to RGB
            data = np.repeat(data, 3, axis=0)
        elif count > 3:
            # Take first 3 (RGB), ignore Alpha/other bands
            data = data[:3, :, :]
        elif count == 2:
            # Pad to 3
            data = np.concatenate([data, data[:1]], axis=0)
            
        # Normalize to 0-1
        data = data.astype(np.float32) / 255.0
        
        return data, src.profile

def match_histograms(source, template):
    """
    Adjust the pixel values of img2 (source) to match the distribution of img1 (template).
    Essential for neutralizing global lighting/color shifts in mining/rural areas.
    """
    # Core logic is pure NumPy, no cv2 dependency required for the math
    src_h, src_w, src_c = source.shape
    matched = np.zeros_like(source)
    
    for i in range(src_c):
        s_values, bin_idx, s_counts = np.unique(source[:, :, i], return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template[:, :, i], return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        matched[:, :, i] = interp_t_values[bin_idx].reshape(src_h, src_w)
        
    return matched

def coregister_images(img1, img2):
    """
     aligns img2 to match img1 using ORB features and RANSAC, then matches histograms.
     img1, img2: numpy arrays (C, H, W) -> need (H, W, C) for OpenCV
    """
    if cv2 is None:
        warnings.warn("OpenCV (cv2) not installed. Skipping co-registration.", ImportWarning)
        return img2

    # Transpose to H,W,C and convert to uint8 for OpenCV
    i1 = (np.transpose(img1, (1, 2, 0)) * 255).astype(np.uint8)
    i2 = (np.transpose(img2, (1, 2, 0)) * 255).astype(np.uint8)
    
    # 1. Histogram Matching (Neutralize Lighting)
    # Match i2 to i1's distribution BEFORE feature matching for better alignment
    i2_matched = match_histograms(i2, i1)
    
    # 2. Convert to grayscale for feature detection
    g1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(i2_matched, cv2.COLOR_RGB2GRAY)
    
    # 3. Detect ORB features
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    
    if des1 is None or des2 is None:
         return np.transpose(i2_matched, (2, 0, 1)).astype(np.float32) / 255.0

    # 4. Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 4:
         return np.transpose(i2_matched, (2, 0, 1)).astype(np.float32) / 255.0

    # 5. Extract points and Find homography
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    if h is None:
        return np.transpose(i2_matched, (2, 0, 1)).astype(np.float32) / 255.0

    # 6. Warp img2 (the histogram-matched version)
    height, width, _ = i1.shape
    aligned_i2 = cv2.warpPerspective(i2_matched, h, (width, height))
    
    return np.transpose(aligned_i2, (2, 0, 1)).astype(np.float32) / 255.0