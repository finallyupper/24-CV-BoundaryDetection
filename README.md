# 24-CV-BoundaryDetection
This repository contains code for **24-Fall Computer Vision Boundary Detection project**. The project dataset is designed to facilitate training and testing of a boundary. 

## Goal
We focus on detecting image edges using classic algorithms, deliberately avoiding machine learning or deep learning techniques. Our goal is to achieve precise boundary detection by combining efficient and interpretable methods that leverage the strengths of traditional filters and advanced post-processing techniques. The approach includes the following core components:
- **Global Filter 1**: Enhances local contrast and reduces noise using CLAHE and Felzenszwalb segmentation.
- **Global Filter 2**: Extracts both global edges and fine details through median blurring, Sobel filtering, and gamma correction.
- **Highlight Filter**: Captures significant edges using Canny edge prediction and contour refinement.
- **Morphological Processor**: Refines edges using erosion and Top-Hat transformations.
- **Skip Connection**: Integrates outputs for robust and detailed edge maps.
  
## Pipeline Overview
This is the pipeline we used for our model:
<div style="text-align: center;">
<img src="./assets/pipeline.png" alt="pipeline" width="70%">
</div> 

## Requirements
Clone this repository:
```
git clone https://github.com/finallyupper/24-CV-BoundaryDetection.git
```
Then install the requirements in your virtual environments. And place the given train and test dataset into `project_data/train` and `project_data/test`.


## Quick Start
To get started with predictions, follow these steps:
1. Open the `main.ipynb` notebook.
2. The notebook demonstrates how to:
    - Load the dataset.
    - Apply filters.
    - Generate predictions using the pipeline.
3. To use a custom combination of filters, modify the total_filter function:

    ```
    def total_filter(images, b1=75, b2=200, limit=60, iter=0):
        f1 = global_filter_1(images, iter=iter)
        f2 = global_filter_2(images)
        f3 = highlight_filter(images, b1, b2, limit)

        f1_f3_norm = get_overlap_lst(f1, f3, "add")
        
        f1f3_conventional = get_overlap_lst(f1_f3_norm, f2, "add", w1=2, w2=8)
        f1f3_conventional_norm = normalize_image(f1f3_conventional)
        
        tophat_out_norm = morph_processor(f1f3_conventional_norm)

        pred_edges = skip_connection(f1f3_conventional_norm, tophat_out_norm)
        return pred_edges 

    ```
## File Structure
- `engine/*`: Utility functions for dataset loading, normalization, and saving outputs as `.npy` files.
- `filters.py`: Traditional edge detection filters (e.g., Sobel, Canny).
- `custom_filters.py`: Five core pipeline components: `global filter 1`, `global filter 2`, `highlight filter`, `morphological processor`, and `skip connection`.
- `evaluate.py`: Tools for evaluation and saving predictions as logs or `.npy` files. (Our prediction file can be downloaded on `./preds/1206_new2_f1sobel_l60_i0_train.npy`).

## Dataset
detection model:
- Annotations: Each image is annotated by three independent annotators.
- Boundary Pixel: 1
- Background Pixel: 0
- Training Set: 200 images (train.zip)
- Test Set: 100 images

## Reference
1. John Canny, A Computational Approach to Edge Detection, IEEE Transactions on Pattern Analysis and Machine Intelligence, (6):679–698, 1986.
2.	Pedro F. Felzenszwalb and Daniel P. Huttenlocher, Efficient Graph-Based Image Segmentation, International Journal of Computer Vision, 59:167–181, 2004.
3.	Nick Kanopoulos et al., Design of an Image Edge Detection Filter Using the Sobel Operator, IEEE Journal of Solid-State Circuits, 23(2):358–367, 1988.
4.	Jean Serra and Pierre Soille, Mathematical Morphology and its Applications to Image Processing, Springer, 2012.
5. Satoshi Suzuki et al., *Topological structural analysis of digitized binary images by border following*, Computer Vision, Graphics, and Image Processing, 30(1):32–46, 1985.
6. Garima Yadav, Saurabh Maheshwari, and Anjali Agarwal, *Contrast limited adaptive histogram equalization based enhancement for real time video system*. In 2014 International Conference on Advances in Computing, Communications and Informatics (ICACCI), pages 2392–2397. IEEE, 2014.
