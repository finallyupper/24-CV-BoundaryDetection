import sys
import os
import numpy as np
from tqdm import tqdm
from cv_project_tools.core.dataset import BoundaryDataset
from cv_project_tools.core import evaluate_boundaries

name = '1206_global2_train'
split = 'train'
num_images = 200


thresholds = 20
apply_thinning = True  

log_file = open(f'/home/yoojinoh/Others/ComputerVision/24-CV-BoundaryDetection/logs/{name}.txt', 'w')

sys.stdout = log_file


root_dir = "/data/yoojinoh/CV/data/"
data_dir = os.path.join(root_dir, "project_data")


os.chdir(root_dir)
print("Current working directory:", os.getcwd())

os.chdir('/home/yoojinoh/Others/ComputerVision/24-CV-BoundaryDetection/cv_project_tools')

# Load the dataset using BoundaryDataset class
dataset = BoundaryDataset(data_dir, split=split)
print(f"Data size: {len(dataset)}")

# Load the saved predictions
predictions = np.load(os.path.join(root_dir, f'/home/yoojinoh/Others/ComputerVision/24-CV-BoundaryDetection/preds/{name}.npy'), allow_pickle=True).item()

# Function to load the prediction for a given sample name
def load_pred(sample_name):
    # Retrieve the prediction for the given sample name from the loaded dictionary
    if sample_name in predictions:
        pred = predictions[sample_name]
        return pred
    else:
        raise KeyError(f"Sample '{sample_name}' not found in the predictions.")
    


sample_names = dataset.sample_names[:num_images]

sample_results, threshold_results, best_result_single, best_result = evaluate_boundaries.pr_evaluation(
    thresholds, sample_names, dataset.load_boundaries, load_pred, apply_thinning=apply_thinning, progress=tqdm
)

print('{:<16}: {:<10.6f}'.format('Best F1', best_result.f1))
print('{:<16}: {:<10.6f}'.format('Best F1 (Single)', best_result_single.f1))
print('{:<16}: {:<10.6f}'.format('AVG F1', (best_result.f1 + best_result_single.f1)/2))

print('[Overall Results]')
print('{:<10} {:<10} {:<10} {:<10}'.format('Recall', 'Precision', 'F1-Score', 'Area PR'))
print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
    best_result.recall, best_result.precision, best_result.f1, best_result.area_pr)
)
print('[Overall Results using Single Threshold]')
print('{:<10} {:<10} {:<10} {:<10}'.format('Threshold', 'Recall', 'Precision', 'F1-Score'))
print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
    best_result_single.threshold, best_result_single.recall, best_result_single.precision, best_result_single.f1)
)
print('[Results Per Image]')
print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('ID', 'Threshold', 'Recall', 'Precision', 'F1-Score'))
for sample_index, res in enumerate(sample_results):
    print('{:<10s} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        res.sample_name, res.threshold, res.recall, res.precision, res.f1))
    
print('[Results Per Threshold]')
print('{:<10} {:<10} {:<10} {:<10}'.format('Threshold', 'Recall', 'Precision', 'F1-Score'))
for thresh_i, res in enumerate(threshold_results):
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        res.threshold, res.recall, res.precision, res.f1))
log_file.close()
