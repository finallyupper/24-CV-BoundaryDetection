import cv2 
import numpy as np 
from engine.dataset import BoundaryDataset
import random 
import matplotlib.pyplot as plt 

def normalize_image(images):
    normalized_image = [(image_array - image_array.min()) / (image_array.max() - image_array.min()) for image_array in images]
    return normalized_image


def get_overlap_lst(test1_lst, test2_lst, type="and", w1=1.0, w2=1.0):
    """Return the overlapping regions of two binary images."""
    overlaps = []
    for test1, test2 in zip(test1_lst, test2_lst):
        if test1.dtype != test2.dtype:
            test2 = test2.astype(test1.dtype) # expect float32 ...? uint8 ok too
        if type =="and":
            overlap = cv2.bitwise_and(w1 * test1, w2 * test2) 
        elif type == "or":
            overlap = cv2.bitwise_or(w1 * test1, w2 * test2)
        elif type == "xor":
            overlap = cv2.bitwise_xor(w1 * test1, w2 * test2)
        elif type == "add":
            overlap = w1 * test1 + w2 * test2
        elif type == "sub":
            overlap = np.maximum(w1 * test1 - w2 * test2, 0)
        overlaps.append(overlap)
    return overlaps


def save_npy(predictions, sample_names, title="test"):
    prediction_npy = {name: array for name, array in zip(sample_names, predictions)}
    np.save(title, prediction_npy)
    print(f"'{title}.npy' saved.")


def get_imgs_boundaries(dataset, selected_sample_names, split='train'):
    images = []
    boundaries_lst = []
    
    for selected_sample_name in selected_sample_names:
        print(f"Processing image: {split}/{selected_sample_name}")
        image = dataset.read_image(selected_sample_name)
        images.append(image)
        if split == "train":
            boundaries = dataset.load_boundaries(selected_sample_name)
            boundaries_lst.append(boundaries )
            fig, ax = plt.subplots(1, len(boundaries) + 1, figsize=(6 * (len(boundaries) + 1), 6))

            ax[0].imshow(image)
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            for i in range(len(boundaries)):
                ax[i + 1].imshow(boundaries[i], cmap='gray')
                ax[i + 1].set_title(f"Boundary {i + 1}")
                ax[i + 1].axis("off")
            plt.tight_layout()
            plt.show()
        elif split =="test":
            plt.imshow(image) 
            plt.title("Original Image")
            plt.axis("off") 
            plt.show()
    return images, boundaries_lst


def load_data(data_path, num_images, split): 
    dataset = BoundaryDataset(data_path, split) 
    selected_sample_names = random.sample(dataset.sample_names, min(num_images, len(dataset.sample_names)))

    images, boundaries = get_imgs_boundaries(dataset, selected_sample_names, split)
    return images, boundaries, selected_sample_names


def get_comparison(images_lst, titles_lst, cmap='gray', base_width=4, base_height=4):
    n_cols = len(images_lst) # # of comparisons
    n_rows = len(images_lst[0]) # # of samples
    
    assert all(len(images) == n_rows for images in images_lst), "All image lists must have the same number of images."
    assert len(images_lst) == len(titles_lst), "The number of image lists and titles must match."

    figsize = (base_width * n_cols, base_height)
    
    for idx in range(n_rows):
        plt.figure(figsize=figsize)
        for col in range(n_cols):
            plt.subplot(1, n_cols, col + 1)
            plt.imshow(images_lst[col][idx], cmap=cmap)
            plt.title(titles_lst[col])
            plt.axis('off')
        plt.tight_layout()
