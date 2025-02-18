import os
import pandas as pd

'''Need to be cleaned'''
NAME_COL = 'fileName'
EMOTION_COL = 'Emotion'
INTENSITY_COL = 'IntensityLabel'
# intensity_df = process_strength(preprocess_config["path"]["preprocessed_path"] + "_unsup/relative_attr/labels")
def process_strength(st_path):
    intensity_dfs = pd.DataFrame()
    for _, speaker in enumerate(os.listdir(st_path)):
        spker_path = os.path.join(st_path, speaker)
        for emotion in os.listdir(spker_path):
            ## if file end with .csv
            if emotion.endswith('.csv'):
                e_path = os.path.join(spker_path, emotion)
                intensity_df = pd.read_csv(e_path)
                intensity_df[NAME_COL] = intensity_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0])
                ### extract the rows that feature paths don't include Neutral
                intensity_df = intensity_df[~intensity_df['feature_paths'].str.contains('Neutral')]
                ## concat it with intensity_dfs
                intensity_df.rename(columns={'X_attr': INTENSITY_COL}, inplace=True)
                # intensity_df.rename(columns={'bin_label': INTENSITY_COL}, inplace=True)
                intensity_dfs = pd.concat([intensity_dfs, intensity_df[[NAME_COL, INTENSITY_COL]]], ignore_index=True)
    ## map 0 to Normal, 1 to High in IntensityLabel
    # intensity_dfs.loc[intensity_dfs[INTENSITY_COL] == 0, INTENSITY_COL] = 'Normal'
    # intensity_dfs.loc[intensity_dfs[INTENSITY_COL] == 1, INTENSITY_COL] = 'High'
    return intensity_dfs
# process_meta_ESD(f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}_N' + f"/train_{dataset_tag}.txt",
#                 f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}' + f"/val_{dataset_tag}_N.txt",
#                 f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}' + f"/val_prompt_{dataset_tag}.txt")
def process_meta_ESD(train_txt, val_txt, out_txt):
    lines = []
    raw_text = []
    with open(val_txt, "r", encoding="utf-8") as f:
        for line in f.readlines():
            n, s, _ , r= line.strip("\n").split("|")
            if n.split('_')[-1] == 'Neutral':
                lines.append(line)
                raw_text.append((s,r))
    for i in [train_txt, val_txt]:
        with open(i, "r", encoding="utf-8") as f:
            for line in f.readlines():
                n, s, _, r= line.strip("\n").split("|")
                if n.split('_')[-1] != 'Neutral':
                    if (s,r) in raw_text:
                        lines.append(line)
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)