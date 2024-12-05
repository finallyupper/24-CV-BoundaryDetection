import cv2 
from filters import * 
import numpy as np 

def custom_filter_1(images):
    images = apply_felz_with_erode(images)
    images = apply_canny(images, 50, 150)
    return images

def custom_filter_3(images):
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    images = apply_median_blur(images, ksize=11)
    images = apply_canny(images, thrs1=75, thrs2=200)
    outputs = []
    for image in images:
        # Find contours
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area and keep the 5 largest
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
        # Create a blank image to plot only contours
        contour_plot = np.zeros_like(image)
        cv2.drawContours(contour_plot, cnts, -1, (255), 2)
        outputs.append(contour_plot)
    return outputs
