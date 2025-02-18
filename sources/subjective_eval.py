import os
import json
import random
from scipy import rand
import torch
import librosa
import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import LabeledDataset
from transformers import Wav2Vec2FeatureExtractor
from objective_eval import process_strength
import text
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, synth_samples, batch2device
from utils.eval_utils import (MCD, WER, ACC, Prompt_GPT, preprocess_english)

'''Subjective evaluation for MOS and PIR'''
device_id = 0
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)
random.seed(44)

NAME_COL = 'fileName'
EMOTION_COL = 'Emotion'
INTENSITY_COL = 'IntensityLabel'

def MOS(device, model, args, configs, vocoder, batchs, result_path):
    preprocess_config, model_config, train_config = configs
    eval_thold = {'Angry': 65.1, 'Happy': 77.5, 'Sad': 64.7, 'Surprise': 77.8}
    data_df = pd.read_csv(args.source, sep='|', header=None, names=['id', 'speaker_id', 'raw_text'], dtype = str)
    prompt_gpt = Prompt_GPT(device, preprocess_config)
    ### Initialize result save path path
    prompt = open("prompt.txt").read()
    raw_text_dict, key_count = {}, 0
    for raw_text in sorted(data_df['raw_text'].unique()):
        raw_text_dict[raw_text] = str(key_count)
        os.makedirs(os.path.join(result_path, str(key_count)), exist_ok=True)
        key_count += 1
    bar = tqdm(total=len(batchs))
    for batch in batchs:
        batch = batch2device(batch, device)       
        path_meta = batch[0][0]
        target_emotion = path_meta.split('_')[1]
        text_id = '_'.join(path_meta.split('_')[1:])
        # print(text_id, target_emotion)
        save_path = os.path.join(result_path, str(raw_text_dict[batch['raw_text'][0]]))
        set1_path = os.path.join(save_path, f'{path_meta}_emt.wav')
        set2_path = os.path.join(save_path, f'{path_meta}_emt_pmt.wav')
        set3_path = os.path.join(save_path, f'{path_meta}_emt_pmt_l.wav')
        prompt_path = os.path.join(save_path, f"{text_id}_prompt.txt")

        prompt_output = prompt_gpt.process(batch, target_emotion, prompt_path)
        if prompt_output == None:
            continue
        utterance_df, word_df, batch = prompt_output
        emt_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'emotion_embeds', 'centroids',  f"{target_emotion}.npy")
        target_emotion_embed_cmt = np.load(emt_embed_path)
        # retrieve prompt
        # Initialize various evaluation settings for synth and objective evaluation
        setting_dict = {
            'set1': {'path': set1_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': target_emotion_embed_cmt},
            'set2': {'path': set2_path, 'global_control': utterance_df, 'local_control': word_df, 'emotion_embed_cmt': target_emotion_embed_cmt},
            'set3': {'path': set3_path, 'global_control': None, 'local_control': word_df, 'emotion_embed_cmt': target_emotion_embed_cmt}}
        with torch.no_grad():
            # Forward
            for k, v in setting_dict.items():
                if not os.path.exists(v['path']):
                    output = model(
                        ids = batch['id'],
                        raw_texts=batch['raw_text'],
                        speakers=batch['speaker_id'],
                        texts=batch['text'],
                        src_lens=batch['text_lens'],
                        max_src_len=batch['max_text_len'],
                        spker_embeds=batch['spker_embed'],
                        wavs=batch['wav'],
                        wav_attn_masks=batch['wav_attn_mask'],
                        emotions=batch['emotion'],
                        intensities=None,
                        global_control=v['global_control'],
                        local_control=v['local_control'],
                        emotion_embed_cmt=v['emotion_embed_cmt'])
                    synth_samples(batch, output, vocoder, model_config, preprocess_config, v['path'], args)
        bar.update(1)

def PIR(device, model, args, configs, vocoder, batchs, result_path):
    preprocess_config, model_config, train_config = configs
    data_df = pd.read_csv(args.source, sep='|', header=None, names=['id', 'speaker_id', 'raw_text'], dtype = str)
    prompt_gpt = Prompt_GPT(device, preprocess_config)
    with open('ref.json', 'r') as f:
        st_ref = json.load(f)
    raw_text_dict, key_count = {}, 0
    for raw_text in data_df['raw_text'].unique():
        raw_text_dict[raw_text] = str(key_count)
        os.makedirs(os.path.join(result_path, str(key_count)), exist_ok=True)
        key_count += 1
    
    bar = tqdm(total=len(batchs))
    for batch in batchs:
        batch = batch2device(batch, device)       
        path_meta = batch[0][0]
        spker, target_emotion = path_meta.split('_')[0], path_meta.split('_')[1]
        text_id = '_'.join(path_meta.split('_')[1:])
        # print(text_id, target_emotion)
        save_path = os.path.join(result_path, str(raw_text_dict[batch['raw_text'][0]]))
        set1_path = os.path.join(save_path, f'{path_meta}_pmt_l_L.wav')
        set2_path = os.path.join(save_path, f'{path_meta}_pmt_l_M.wav')
        set3_path = os.path.join(save_path, f'{path_meta}_pmt_l_H.wav')
        prompt_path = os.path.join(save_path, f"{text_id}_prompt.txt")

        prompt_output = prompt_gpt.process(batch, target_emotion, prompt_path)
        if prompt_output == None:
            continue
        utterance_df, word_df, batch = prompt_output
        emt_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'emotion_embeds', 'centroids',  f"{target_emotion}.npy")
        target_emotion_embed_cmt = np.load(emt_embed_path)
        L_id, M_id, H_id = st_ref[spker][target_emotion]['Low'], st_ref[spker][target_emotion]['Medium'], st_ref[spker][target_emotion]['High']
        st_L_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'intensity_embeds', spker,  f"{L_id}.npy")
        st_M_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'intensity_embeds', spker,  f"{M_id}.npy")
        st_H_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'intensity_embeds', spker,  f"{H_id}.npy")
        # retrieve prompt
        # Initialize various evaluation settings for synth and objective evaluation
        setting_dict = {
            'set1': {'path': set1_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': np.load(st_L_embed_path)},
            'set2': {'path': set2_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': np.load(st_M_embed_path)},
            'set3': {'path': set3_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': np.load(st_H_embed_path)}}
        with torch.no_grad():
            # Forward
            for _, v in setting_dict.items():
                if not os.path.exists(v['path']):
                    output = model(
                        ids = batch['id'],
                        raw_texts=batch['raw_text'],
                        speakers=batch['speaker_id'],
                        texts=batch['text'],
                        src_lens=batch['text_lens'],
                        max_src_len=batch['max_text_len'],
                        spker_embeds=batch['spker_embed'],
                        wavs=batch['wav'],
                        wav_attn_masks=batch['wav_attn_mask'],
                        emotions=batch['emotion'],
                        intensities=None,
                        global_control=v['global_control'],
                        local_control=v['local_control'],
                        emotion_embed_cmt=v['emotion_embed_cmt'],
                        intensity_embed_cmt=v['intensity_embed_cmt'])
                    synth_samples(batch, output, vocoder, model_config, preprocess_config, v['path'], args)
        bar.update(1)

def PIR_ESD(device, model, args, configs, vocoder, batchs, st_df, result_path):
    preprocess_config, model_config, train_config = configs
    data_df = pd.read_csv(args.source, sep='|', header=None, names=['id', 'speaker_id', 'text', 'raw_text'], dtype = str)
    
    prompt_gpt = Prompt_GPT(device, preprocess_config)
    ### Initialize result save path path
    # prompt = open("prompt.txt").read()
    raw_text_dict, key_count = {}, 0
    # for raw_text in data_df['raw_text'].unique():
    #     raw_text_dict[raw_text] = str(key_count)
    #     # os.makedirs(os.path.join(result_path, str(key_count)), exist_ok=True)
    #     key_count += 1
    angry_texts = random.choices(data_df[data_df['id'].str.contains('Angry')]['raw_text'].unique(), k = 4)
    happy_texts = random.choices(data_df[data_df['id'].str.contains('Happy')]['raw_text'].unique(), k = 4)
    sad_texts = random.choices(data_df[data_df['id'].str.contains('Sad')]['raw_text'].unique(), k = 4)
    surprise_texts = random.choices(data_df[data_df['id'].str.contains('Surprise')]['raw_text'].unique(), k = 4)
    cdd_texts = {'Angry': angry_texts, 'Happy': happy_texts, 'Sad': sad_texts, 'Surprise': surprise_texts}
    # cdd_texts = {'Angry': angry_texts, 'Sad': sad_texts}
    for k,v in cdd_texts.items():
        for raw_text in v:
            raw_text_dict[raw_text] = k+'_'+str(key_count)
            os.makedirs(os.path.join(result_path, k+'_'+str(key_count)), exist_ok=True)
            key_count += 1
    st_pointers = {'Lvl_1': '0011_001130_Sad', 'Lvl_2': '0015_001131_Sad', 'Lvl_3': '0019_000927_Happy'}
    bar = tqdm(total=len(batchs))
    for batch in batchs:
        batch = batch2device(batch, device)    
        if batch['raw_text'][0] not in raw_text_dict.keys():
            bar.update(1)
            continue
        path_meta = batch[0][0].split('_')
        text_id = raw_text_dict[batch['raw_text'][0]]
        # if 'Sad' not in text_id:
        #     bar.update(1)
        #     continue
        raw_text = batch['raw_text'][0]
        unmap_spker_id, emotion = path_meta[0], path_meta[2]
        if emotion != text_id.split('_')[0]:
            bar.update(1)
            continue
        # os.system(f'cp {preprocess_config["path"]["raw_path"]}/{wav_id}_Neutral.wav {os.path.join(result_path, str(raw_text_dict[batch["raw_text"][0]]))}')
        save_path = os.path.join(result_path, f'{text_id}')
        set1_path = os.path.join(save_path, f'{batch[0][0]}_pmt_l_L.wav')
        set2_path = os.path.join(save_path, f'{batch[0][0]}_pmt_l_M.wav')
        set3_path = os.path.join(save_path, f'{batch[0][0]}_pmt_l_H.wav')
        prompt_path = os.path.join(save_path, f"{text_id}_prompt.txt")
        emt_embed_path = os.path.join(
            os.path.dirname(result_path), 
            'emotion_embeds', unmap_spker_id,  f"{batch[0][0]}.npy")
        # if not os.path.exists(emt_embed_path):
        #     print(f'{emt_embed_path} not found')
        #     continue
        emotion_embed_cmt = np.load(emt_embed_path)
        
        prompt_output = prompt_gpt.process(batch, emotion, prompt_path)
        if prompt_output == None:
            continue
        _, word_df, batch = prompt_output
        # Initialize various evaluation settings for synth and objective evaluation
        setting_dict = {
            'set1': {'path': set1_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': emotion_embed_cmt, 'intensity_embed_cmt': 'Lvl_1'},
            'set2': {'path': set2_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': emotion_embed_cmt, 'intensity_embed_cmt': 'Lvl_2'},
            'set3': {'path': set3_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': emotion_embed_cmt, 'intensity_embed_cmt': 'Lvl_3'}
            }
        with open(save_path + '/phonemes.txt', 'w') as f:
            f.write(' '.join(word_df['word'].tolist()))

        for _, v in setting_dict.items():
            spker = st_pointers[v['intensity_embed_cmt']].split('_')[0]
            st_id = st_pointers[v['intensity_embed_cmt']]
            st_path = os.path.join(
                    os.path.dirname(result_path), 
                    'intensity_embeds', spker, f"{st_id}.npy")
            pe_path = os.path.join(save_path, os.path.basename(v['path']).replace('.wav', '.json'))
            if not os.path.exists(st_path):
                print(f'{st_path} not found')
                continue
            v['intensity_embed_cmt'] = np.load(st_path)
            if os.path.exists(v['path']):
                print(v['path'], ' exists')
                continue
            with torch.no_grad():
                # Forward
                output = model(
                    ids = batch['id'],
                    raw_texts=batch['raw_text'],
                    speakers=batch['speaker_id'],
                    texts=batch['text'],
                    src_lens=batch['text_lens'],
                    max_src_len=batch['max_text_len'],
                    spker_embeds=batch['spker_embed'],
                    wavs=batch['wav'],
                    wav_attn_masks=batch['wav_attn_mask'],
                    emotions=batch['emotion'],
                    intensities=None,
                    global_control=v['global_control'],
                    local_control=v['local_control'],
                    emotion_embed_cmt=v['emotion_embed_cmt'],
                    intensity_embed_cmt=v['intensity_embed_cmt'])
                ## save output.p_predictions
                synth_samples(batch, output, vocoder, model_config, preprocess_config, v['path'], args)
                ## save pitch and energy
                pitch_pred = output.p_predictions['pitch_pred'].cpu().squeeze().tolist()
                f0_denorm = output.p_predictions['f0_denorm'].cpu().squeeze().tolist()
                energy_pred = output.e_predictions.cpu().squeeze().tolist()
                pe_dict = {'pitch': pitch_pred, 'f0_denorm': f0_denorm, 'energy': energy_pred}
                with open(pe_path, 'w') as f:
                    json.dump(pe_dict, f)
        bar.update(1)

# python subjective_eval.py --source val_mos.txt --restore_step 450000 --mode batch --checkpoint ESD --label emotion
# python subjective_eval.py --source val_PIR.txt --restore_step 450000 --mode batch --checkpoint ESD --label intensity
# python subjective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --mode batch --checkpoint ESD --label intensity
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], required=True, help="Synthesize a whole dataset or a single sentence")
    parser.add_argument("--source", type=str, default=None, help="path to a source file with format like train.txt and val.txt, for batch mode only")
    parser.add_argument("--text", type=str, default=None, help="raw text to synthesize, for single-sentence mode only")
    parser.add_argument("--speaker_id", type=str, default="p225", help="speaker ID for multi-speaker synthesis, for single-sentence mode only")
    parser.add_argument("--checkpoint", type=str, required=True, help="name of dataset")
    parser.add_argument("--label", type=str, default=None, help="emotion or intensity")
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.checkpoint)
    configs = (preprocess_config, model_config, train_config)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
    
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    dataset_tag = "unsup" if learn_alignment else "sup"
    pitch_type = preprocess_config['preprocessing']['pitch']['pitch_type']
    
    args.checkpoint = os.path.join(os.path.dirname(train_config['path']), 
                                       f'{args.checkpoint}_{dataset_tag}_{pitch_type}_wi')
    result_path = os.path.join(
        args.checkpoint, 
        'result',
        str(args.restore_step), 
        'PIR_eval')
    os.makedirs(result_path, exist_ok=True)
    # Set Device
    torch.manual_seed(train_config["seed"])
    print("Device of CompTransTTS:", device)
    # Get model
    # process_meta_ESD(f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}_N' + f"/train_{dataset_tag}.txt",
    #                 f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}' + f"/val_{dataset_tag}_N.txt",
    #                 f'{preprocess_config["path"]["preprocessed_path"]}_{dataset_tag}' + f"/val_prompt_{dataset_tag}.txt")

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    # Get dataset
    # for prompt dataset, batch size should be 1
    dataset = LabeledDataset(args.source, preprocess_config, model_config)
    # model = None
    batchs = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate_fn)
    if args.label == 'emotion':
        model = get_model(args, configs, device, train=False, labels_e=dataset.emotion_set, labels_i=False)
        # st_df = None
    elif args.label == 'intensity':
        model = get_model(args, configs, device, train=False, labels_e=dataset.emotion_set, labels_i=True)
    st_df = process_strength(preprocess_config["path"]["preprocessed_path"] + f"_{dataset_tag}/relative_attr/labels")
    PIR_ESD(device, model, args, configs, vocoder, batchs, st_df, result_path)
