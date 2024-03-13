import os
import numpy as np

emotion_dict = {'Anger':0, 'Excited':1, 'Frustration':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
data_src_dit = 'Audio_Data/raw_data'
preprocess_root_dit = 'Audio_Data/preprocessed_data/IMOCap/relative_attr'
if not os.path.exists(os.path.dirname(os.path.dirname(preprocess_root_dit))):
    os.makedirs(os.path.dirname(os.path.dirname(preprocess_root_dit)))
if not os.path.exists(os.path.dirname(preprocess_root_dit)):
    os.makedirs(os.path.dirname(preprocess_root_dit))
if not os.path.exists(preprocess_root_dit):
    os.makedirs(preprocess_root_dit)

datadict_path = os.path.join(preprocess_root_dit, "datadict_IMOCap.npy")
weights_dit = os.path.join(preprocess_root_dit, "weights")
os.makedirs(weights_dit, exist_ok=True)
results_dit = os.path.join(preprocess_root_dit, "labels")
os.makedirs(results_dit, exist_ok=True)
cat_ordering = np.array([1,1,1,1,0,1,1])