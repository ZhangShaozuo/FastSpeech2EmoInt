import os
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

opensmile_path = 'Audio_Data/opensmile-3.0-linux-x64/bin/SMILExtract'
config_path = 'Audio_Data/opensmile-3.0-linux-x64/config/is09-13/IS09_emotion.conf'

# def preprocess():
#     count = 0
#     error_list = []
#     data_dict = {'class_labels': [], 'speakers': [], 'feature_paths': [], 'feat':[]}
#     for _ , speaker in enumerate(tqdm(os.listdir(data_path))):
#         spk_out_dir = os.path.join(save_path, speaker)
#         spk_in_path = os.path.join(data_path, speaker)
#         os.makedirs(spk_out_dir, exist_ok=True)
#         for _, emotion in enumerate(os.listdir(spk_in_path)):
#             e_in_path = os.path.join(spk_in_path, emotion)
#             e_out_path = os.path.join(spk_out_dir, emotion)
#             if os.path.isdir(e_in_path):
#                 os.makedirs(e_out_path, exist_ok=True)
#                 for _, wav_name in enumerate(os.listdir(e_in_path)):
#                     if wav_name.endswith('.wav'):
#                         wavfile_path = os.path.join(e_in_path, wav_name)
#                         feature_path = os.path.join(e_out_path, wav_name.replace('.wav', '.csv'))
#                         try:
#                             os.system(opensmile_path + ' -C ' + config_path + ' -I ' + wavfile_path + ' -csvoutput ' + feature_path + ' -instname ' + feature_path[-15:-4])
#                             data_dict['class_labels'].append(emotion_dict[emotion])
#                             data_dict['speakers'].append(speaker)
#                             data_dict['feature_paths'].append(feature_path)
#                             feat = pd.read_csv(feature_path, sep=';').values[0][2:]
#                             feat_norm = (feat-np.min(feat))/(np.max(feat)-np.min(feat))
#                             data_dict['feat'].append(feat_norm)
#                         except:
#                             error_list.append(wav_name)
#                             count += 1

#     data_dict['class_labels'] = np.array(data_dict['class_labels'])
#     data_dict['speakers'] = np.array(data_dict['speakers'])
#     data_dict['feat'] = np.array(data_dict['feat'])

#     np.save(datadict_path, data_dict)
#     print('error num: ',count)  # This should be 0
#     print('error list: ', error_list)

def preprocess():
    count = 0
    error_list = []
    data_dict = {'class_labels': [], 'speakers': [], 'feature_paths': [], 'feat':[]}
    # split = {}
    for _ , speaker in enumerate(tqdm(os.listdir(data_path))):
        spk_out_dit = os.path.join(save_path, speaker)
        spk_in_dit = os.path.join(data_path, speaker)
        os.makedirs(spk_out_dit, exist_ok=True)
        # split[speaker] = {}
        for _, wav_name in enumerate(os.listdir(spk_in_dit)):
            if wav_name.endswith('.wav'):
                wavfile_path = os.path.join(spk_in_dit, wav_name)
                feature_path = os.path.join(spk_out_dit, wav_name).replace('.wav', '.csv')
                
                instname = os.path.basename(feature_path).split('.')[0]
                # wav_id = '_'.join(os.path.basename(feature_path).split('.')[0].split('_')[:2])
                emotion = os.path.basename(wavfile_path).split('.')[0].split('_')[-1]
                # feature_path = os.path.join(
                #     os.path.dirname(feature_path), emotion, instname + '.csv')
                try:
                    os.system(opensmile_path + ' -C ' + config_path + ' -I ' + wavfile_path + ' -csvoutput ' + feature_path + ' -instname ' + instname)
                    data_dict['class_labels'].append(emotion_dict[emotion])
                    data_dict['speakers'].append(speaker)
                    data_dict['feature_paths'].append(feature_path)
                    feat = pd.read_csv(feature_path, sep=';').values[0][2:]
                    feat_norm = (feat-np.min(feat))/(np.max(feat)-np.min(feat))
                    data_dict['feat'].append(feat_norm)
                except:
                    error_list.append(wav_name)
                    count += 1

    data_dict['class_labels'] = np.array(data_dict['class_labels'])
    data_dict['speakers'] = np.array(data_dict['speakers'])
    data_dict['feat'] = np.array(data_dict['feat'])

    np.save(datadict_path, data_dict)
    print('error num: ',count)  # This should be 0
    print('error list: ', error_list)


# python learned_rank_func/preprocess_os.py --dataset IMOCap
# python learned_rank_func/preprocess_os.py --dataset ESD
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ESD', help='ESD or IMOCap')
    args = parser.parse_args()

    if args.dataset == 'ESD':
        from configs.esd_config import *
    elif args.dataset == 'IMOCap':
        from configs.imocap_config import *
    data_path = data_src_dit +f'/{args.dataset}'
    save_path = data_src_dit + f'/{args.dataset}_OS'
    os.makedirs(save_path, exist_ok=True)
    preprocess()


    



