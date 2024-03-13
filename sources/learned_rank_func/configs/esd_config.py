import os
import numpy as np
emotion_dict = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
data_src_dit = 'Audio_Data/raw_data'
preprocess_root_dit = 'Audio_Data/preprocessed_data/ESD/relative_attr' ## later change to IMOCap
if not os.path.exists(os.path.dirname(os.path.dirname(preprocess_root_dit))):
    os.makedirs(os.path.dirname(os.path.dirname(preprocess_root_dit)))
if not os.path.exists(os.path.dirname(preprocess_root_dit)):
    os.makedirs(os.path.dirname(preprocess_root_dit))
if not os.path.exists(preprocess_root_dit):
    os.makedirs(preprocess_root_dit)
datadict_path = os.path.join(preprocess_root_dit, "features.npy")
weights_dit = os.path.join(preprocess_root_dit, "weights")
os.makedirs(weights_dit, exist_ok=True)
results_dit = os.path.join(preprocess_root_dit, "results")
os.makedirs(results_dit, exist_ok=True)
# cat_ordering = np.array([0,1])
cat_ordering = np.array([1,1,0,1,1])
# # sorted_cat_idx = np.argsort(cat_ordering)