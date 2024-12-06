import cv2 
from filters import * 
from engine.utils import normalize_image
import numpy as np 

def global_filter_1(images, iter=0)->list: 
    """
    Global Filter 1
    This filter enhances edge features in the image using CLAHE, Felzenszwalb segmentation, and Sobel operator.
    Parameters:
        images (list): List of input images.
        iter (int): Number of iterations for Felzenszwalb segmentation (default 0).

    Returns:
        list: Filtered and normalized grayscale images highlighting edges.
    """
    images = apply_clahe(images, tileGridSize=(2, 2))
    images = apply_felz(images, iterations=iter)    
    images = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in images]
    images = apply_sobel(images)[2] 
    images = normalize_image(images)
    return images

def highlight_filter(images, thrs1=75, thrs2=200, limit=50)->list: 
    """
    Highlight Filter:
    This filter emphasizes key edge features by detecting contours in the image.

    Parameters:
        images (list): List of input images.
        thrs1 (int): Lower threshold for Canny edge detection.
        thrs2 (int): Upper threshold for Canny edge detection.
        limit (int): Number of largest contours to retain.
    Returns:
        list: Images with prominent contours highlighted.
    """
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    images = apply_median_blur(images, ksize=11)
    images = apply_canny(images, thrs1=thrs1, thrs2=thrs2)
    outputs = []
    for image in images:
        # Find contours
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area and keep the 5 largest
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:limit]
        # Create a blank image to plot only contours
        contour_plot = np.zeros_like(image)
        cv2.drawContours(contour_plot, cnts, -1, (255), 2)
        outputs.append(contour_plot)
    outputs = normalize_image(outputs)
    return outputs

def global_filter_2(images)->list: 
    """
    Global Filter 2:
    This filter processes images to enhance edges using median blur, Sobel operator, and gamma correction.

    Parameters:
        images (list): List of input images.
    Returns:
        list: Filtered and normalized images with enhanced edges.
    """
    test_median = apply_median_blur(images, ksize=7)
    test_median_sobel = apply_sobel(test_median, ksize= 3)[2]
    test_median_sobel_gray = convert_to_grayscale(test_median_sobel)
    test_median_sobel_gray_gamma = apply_gamma_correction(test_median_sobel_gray, gamma=1.1)
    test_median_sobel_gray_gamma_normalized = normalize_image(test_median_sobel_gray_gamma)
    return test_median_sobel_gray_gamma_normalized 


def morph_processor(images)->list:
    """
    Morphological Processor:
    This function applies morphological operations to refine edges.
    - Erosion shrinks boundaries of objects to remove noise.
    - Top-hat operation highlights fine structures in the image.
    - Thresholding isolates high-intensity regions.

    Parameters:
        images (list): List of input images.

    Returns:
        list: Morphologically processed and normalized images.
    """
    erosion = apply_erosion(images) 
    tophat = apply_tophat(erosion) 

    tophat_255 = norm_to_255(tophat) 
    thresholded_image = apply_custom_threshold(tophat_255, thresh=254) 

    tophat_out = [(i1 + i2) for (i1, i2) in zip(tophat_255, thresholded_image)] 
    tophat_out_norm = normalize_image(tophat_out)

    return tophat_out_norm

def skip_connection(f1f3_conventional_norm, tophat_out_norm, w1=0.6, w2=1.0)->list: 
    """
    Skip Connection:
    This function combines outputs from two filters using weighted addition.
    - Weighted contributions from `f1f3_conventional_norm` and `tophat_out_norm` are merged.
    - Skip connections help preserve information from different stages of processing.

    Parameters:
        f1f3_conventional_norm (list): Filtered output from the combination of Global filters and the highlight filter.
        tophat_out_norm (list): Filtered output from morphological processor.
        w1 (float): Weight for F1 and F3 output (default 0.6).
        w2 (float): Weight for morphological processor output (default 1.0).

    Returns:
        list: Combined and normalized edge-detected images.
    """
    pred_edges = [(w1*i1 + w2*i2) for (i1, i2) in zip(f1f3_conventional_norm, tophat_out_norm)] 
    pred_edges = normalize_image(pred_edges)
    return pred_edges 

