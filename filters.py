import cv2 
import numpy as np 
from skimage.color import label2rgb
from skimage import data, color, graph
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2lab, label2rgb
from skimage.segmentation import slic, mark_boundaries, felzenszwalb

def apply_median_blur(images, ksize=5):
    medians = []
    for image in images:
        median = cv2.medianBlur(image, ksize) # sigmaY안적으로 sigmaX와 동일 처리 
        medians.append(median)
    return medians 

def apply_sobel(images, ksize=3):
    grad_x_lst, grad_y_lst, sobels = [], [], []
    for image in images:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)  # Gradient in x direction 
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)  # Gradient in y direction

        # Convert gradients to absolute values and then to 8-bit
        grad_x = cv2.convertScaleAbs(grad_x) # unsigned integer values
        grad_y = cv2.convertScaleAbs(grad_y)

        # Combine the gradients to get the edge map
        sobel_edge = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        grad_x_lst.append(grad_x)
        grad_y_lst.append(grad_y)
        sobels.append(sobel_edge)
    return grad_x_lst, grad_y_lst, sobels 

def apply_felz(images, scale=400, sigma=2, min_size=60, kernel_size=(3, 3), iterations=0):
    felz_images = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # Rectangular kernel

    for image in images:
        # Step 1: Felzenswalb segmentation
        felz_segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
        segmented_image = label2rgb(felz_segments, image=image, kind='avg')

        # Step 2: Convert to uint8
        segmented_image_uint8 = (segmented_image * 255).astype(np.uint8) if segmented_image.max() <= 1 else segmented_image.astype(np.uint8)

        # Step 3: Apply cv2.erode using the kernel
        eroded_image = cv2.erode(segmented_image_uint8, kernel, iterations=iterations)

        # Step 4: Add to the result list
        felz_images.append(eroded_image)

    return felz_images


def apply_canny(images, thrs1, thrs2):
    canny_edges = []
    for image in images:
        canny_edge = cv2.Canny(image, threshold1=thrs1, threshold2=thrs2)
        canny_edges.append(canny_edge)
    return canny_edges 


def convert_to_grayscale(images):
    grayscale_images = []
    for image in images:
        if len(image.shape) == 3:  # Check if the image is 3D
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # Already 2D
            grayscale = image
        grayscale_images.append(grayscale)
    return grayscale_images

def apply_gamma_correction(image_scaled_lst, gamma = 0.5 ):
    outputs = []
    for image_scaled in image_scaled_lst:
        image_gamma_corrected = np.power(image_scaled / 255.0, gamma) * 255
        image_gamma_corrected = image_gamma_corrected.astype(np.uint8)
        outputs.append(image_gamma_corrected)
    return outputs 

def apply_erosion(images, ksize=(2, 2)):
    outputs = [] 
    kernel = np.ones(ksize, np.uint8)

    for image in images:
        erosion = cv2.erode(image, kernel)
        outputs.append(erosion)

    return outputs 

def apply_tophat(erosion_images, ksize=(5, 5)):
    outputs = [] 
    kernel = np.ones(ksize, np.int8) 
    for image in erosion_images:
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        outputs.append(tophat) 
    return outputs

def apply_threshold(images, thresh=100, max=255, type=cv2.THRESH_BINARY): # 0~255로 사전 변환 필요
    outputs = [] 
    for image in images:
        _, adjusted = cv2.threshold(image, thresh, max, type)
        outputs.append(adjusted) 
    return outputs

def apply_custom_threshold(images, thresh=100, max=255):
    """
    Apply a threshold operation where pixels above `thresh` are set to `max`,
    and pixels below `thresh` retain their original value.
    """
    outputs = []
    for image in images:
        # 조건에 따라 픽셀 값 변환
        adjusted = np.where(image > thresh, max, image)
        outputs.append(adjusted)
    return outputs

def apply_custom_threshold2(images, thresh=100, factor=2.0):
    """
    Apply a custom operation:
    - If a pixel value is below `thresh`, multiply it by `factor`.
    - Otherwise, keep the pixel value unchanged.
    
    Parameters:
    - images: List of images (numpy arrays).
    - thresh: Threshold value.
    - factor: Multiplication factor for pixels below `thresh`.
    
    Returns:
    - List of processed images.
    """
    outputs = []
    for image in images:
        # 조건에 따라 픽셀 값 변환
        adjusted = np.where(image < thresh, image * factor, image)
        
        # 픽셀 값이 255를 초과하지 않도록 클리핑
        adjusted = np.clip(adjusted, 0, 255).astype(image.dtype)
        outputs.append(adjusted)
    return outputs

def norm_to_255(images):
    outputs = [] 
    for img in images:
        if np.max(img) <= 1:
            modified = (img * 255).astype(np.uint8) 
            outputs.append(modified) 
        else:
            print("CHECK RANGE OF THE IMAGE!")
    return outputs 

def apply_laplacian(images, ksize, border_type):
    results = [] 
    for image in images:
        edge = cv2.Laplacian(image, -1, ksize=ksize, borderType=border_type)
        results.append(edge)
    return results 


def apply_highboost(images, boost_factor=3):
    highboost_images = []
    for image in images:
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        highboost_mask = cv2.subtract(image, blurred_image)
        highboost_image = cv2.addWeighted(image, boost_factor, highboost_mask, 1, 0)
        highboost_images.append(highboost_image)
    return highboost_images

#### NOT IMPORTED ####

def apply_kmeans(images,k=3, max_iter=100, eps=0.2):
    # BGR image
    def bgr_to_pixels(images):
        results = [] 
        for image in images:
            assert len(image.shape) == 3
            pixels = image.reshape((-1, 3)) 
            pixels = np.float32(pixels) 
            results.append(pixels) 
        return results 
    pixels_lst = bgr_to_pixels(images)
    segmented_images = []
    for idx, pixels in enumerate(pixels_lst):
        # Define criteria and apply KMeans
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iter, eps)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS) # 
        # Convert centers to uint8 and map labels to corresponding pixel values
        centers = np.uint8(centers) 
        segmented_image = centers[labels.flatten()] 
        segmented_image = segmented_image.reshape(images[idx].shape)
        segmented_images.append(segmented_image)
    return segmented_images # BHR images

def apply_mean_shift(images, quantile=0.1, n_samples=500):
    """
    Mean shift clustering 
    (No need for k as hyperparam)
    """
    mean_shift_images = [] 
    def bgr_to_pixels(images):
        results = [] 
        for image in images:
            assert len(image.shape) == 3
            pixels = image.reshape((-1, 3)) 
            pixels = np.float32(pixels) 
            results.append(pixels) 
        return results 

    pixels_lst = bgr_to_pixels(images) 
    for idx, pixels in enumerate(pixels_lst):
        # Estimate bandwidth for Mean Shift
        bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=n_samples) 
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True) 

        # Fit and predict clusters
        mean_shift.fit(pixels)
        labels = mean_shift.labels_
        cluster_centers = mean_shift.cluster_centers_ # peak coordinate    
        #print("[Mean Shift] # Unique Labels:", len(np.unique(labels))) # debug

        # Map labels to corresponding pixel values
        cluster_centers = np.uint8(cluster_centers)
        mean_shift_image = cluster_centers[labels.flatten()]
        mean_shift_image = mean_shift_image.reshape(images[idx].shape)
        mean_shift_images.append(mean_shift_image) 

    return mean_shift_images

def apply_ncut(images, n_segs=400, compactness=30, sigma=1):
    """
    Graph cut
    - Apply slic (simple linear iterative clustering) -> superpixel algorithm 
    - rag_mean_color(Region Adjacency Graph) 
    - cut_normalized
    """
    # bgr images as inputs
    seg_images = []
    for image in images:
        # Apply SLIC superpixel segmentation
        segments = slic(image, n_segments=n_segs, compactness=compactness, sigma=sigma)
        # Build Region Adjacency Graph (RAG)
        rag = graph.rag_mean_color(image, segments, mode='similarity')
        # Apply Normalized Cut
        labels = graph.cut_normalized(segments, rag)
        print("[Graph Cut] # Unique Labels:", len(np.unique(labels))) 
        segmented_image = label2rgb(labels, image, kind='avg', bg_label=0)
        seg_images.append(segmented_image) 
    return seg_images # BGR image 


def apply_clahe(images, clipLimit=2.0, tileGridSize=(8, 8), type='color'):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Params:
    - images : List of images. For 'color', use RGB images. For 'gray', use grayscale images.
    - clipLimit : Threshold for contrast limiting.
    - tileGridSize : Size of grid for CLAHE. Input as a tuple (x, y).
    - type : 'gray' for grayscale images, 'color' for RGB images.
    
    Returns:
    - List of images with CLAHE applied.
    """
    if type == 'gray':
        gray_clahes = []
        for image in images:
            # Ensure the image is already grayscale
            if len(image.shape) == 3:
                raise ValueError("Input images must be grayscale for 'gray' type.")
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            gray_clahe = clahe.apply(image)
            gray_clahes.append(gray_clahe)
        return gray_clahes
    
    elif type == 'color':
        rgb_image_clahes = []
        for image in images:
            # Ensure the image is RGB
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Input images must be RGB for 'color' type.")
            b, g, r = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)
            image_clahe = cv2.merge((b_clahe, g_clahe, r_clahe))
            rgb_image_clahe = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
            rgb_image_clahes.append(rgb_image_clahe)
        return rgb_image_clahes

    else:
        raise ValueError("Invalid type specified. Use 'gray' or 'color'.")

def apply_otsu(images):
    otsu_threshold_values, otsu_threshs = [], []
    for i, image in enumerate(images):
        otsu_threshold_value, otsu_thresh = \
            cv2.threshold(image,  
                          0, # automatic이라 thres안적어도 무방
                          255,  # >T이면 255 할당 (binary image)
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU) # binary thesholding + otsu thresholding 적용(automa)
        otsu_threshold_values.append(otsu_threshold_value)
        otsu_threshs.append(otsu_thresh) 
    return otsu_threshold_values, otsu_threshs 


def apply_adap_gaussian_thrs(images):
    adaptive_thresh_gaussians = []
    for image in images:
        adaptive_thresh_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        adaptive_thresh_gaussians.append(adaptive_thresh_gaussian)
    return adaptive_thresh_gaussians 

def apply_avg_blur(images, ksize=(5, 5)):
    avg_gaussians = []
    for image in images:
        avg_gaussian = cv2.blur(image, ksize)
        avg_gaussians.append(avg_gaussian)
    return avg_gaussians 

def apply_gaussian_blur(images, ksize=(5, 5)):
    gaussians = []
    for image in images:
        gaussian = cv2.GaussianBlur(image, ksize, 1) # sigmaY안적으로 sigmaX와 동일 처리 
        gaussians.append(gaussian)
    return gaussians 

