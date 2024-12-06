import cv2 
import numpy as np 
from skimage.color import label2rgb
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb

def apply_median_blur(images, ksize=5):
    medians = []
    for image in images:
        median = cv2.medianBlur(image, ksize) 
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

def apply_threshold(images, thresh=100, max=255, type=cv2.THRESH_BINARY): 
    outputs = [] 
    for image in images:
        _, adjusted = cv2.threshold(image, thresh, max, type)
        outputs.append(adjusted) 
    return outputs

def apply_custom_threshold(images, thresh=100, max=255):
    outputs = []
    for image in images:
        adjusted = np.where(image > thresh, max, image)
        outputs.append(adjusted)
    return outputs

def apply_custom_threshold2(images, thresh=100, factor=2.0):
    outputs = []
    for image in images:
        adjusted = np.where(image < thresh, image * factor, image)
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
