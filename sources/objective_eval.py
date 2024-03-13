import os
import json
import torch
import librosa
import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import LabeledDataset
from transformers import Wav2Vec2FeatureExtractor
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, synth_samples, batch2device
from utils.eval_utils import (MCD, WER, ACC, Prompt_GPT, preprocess_english)


device_id = 1
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)

NAME_COL = 'fileName'
EMOTION_COL = 'Emotion'
INTENSITY_COL = 'IntensityLabel'

def ECA(device, model, args, configs, batchs, result_path):
    '''This function computes'''
    preprocess_config, model_config, _ = configs
    data_df = pd.read_csv(args.source, sep='|', header=None, names=['id', 'speaker_id', 'text', 'raw_text'], dtype = str)
    
    acc = ACC(id2label = model.emotion_encoder.config.id2label, result_path=result_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_config['emotion_encoder']['model_ckpt'])
    sampling_rate = feature_extractor.sampling_rate
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    
    raw_text_dict, key_count = {}, 0
    for raw_text in data_df['raw_text'].unique():
        raw_text_dict[raw_text] = str(key_count)
        os.makedirs(os.path.join(result_path, str(key_count)), exist_ok=True)
        key_count += 1

    bar = tqdm(total=len(batchs))
    for batch in batchs:
        batch = batch2device(batch, device)       
        path_meta = batch[0][0].split('_')
        unmap_spker_id, wav_id, emotion = path_meta[0], '_'.join(path_meta[0:2]), path_meta[2]
        if emotion != 'Neutral':
            bar.update(1)
            continue
        for target_emotion in preprocess_config['emotion_labels']:
            save_path = os.path.join(result_path, str(raw_text_dict[batch['raw_text'][0]]))
            set1_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_no_prompt.wav')
            set2_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word-utter_prompt.wav')
            set3_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word_prompt.wav')
            # check reference target sample exists 
            target_df = data_df[data_df['id'].str.contains(target_emotion)
                    & (data_df['raw_text'] == batch['raw_text'][0])
                    & (data_df['speaker_id'] == unmap_spker_id)]
            if len(target_df) == 0:
                print(f'Conditions {target_emotion}, {batch["raw_text"][0]}, {unmap_spker_id} not found in data_df')
                continue
            setting_dict = {'set1': {'path': set1_path},
                            'set2': {'path': set2_path},
                            'set3': {'path': set3_path}}            
            with torch.no_grad():
                for k, v in setting_dict.items():
                    if os.path.exists(v['path']):
                        wav, _ = librosa.load(v['path'], sr = sampling_rate)
                        wav = wav / max(abs(wav)) * max_wav_value
                        feature = feature_extractor(wav, 
                                                        sampling_rate=sampling_rate, 
                                                        return_tensors="pt", 
                                                        return_attention_mask=True)
                        input_values = feature.input_values.to(device)
                        attention_mask = feature.attention_mask.to(device)
                        target_emotion_id = torch.tensor([model.emotion_encoder.config.label2id[target_emotion]]).to(device)
                        
                        output = model.emotion_encoder(
                            input_values,
                            attention_mask=attention_mask,
                            labels = target_emotion_id)
                        
                        pred = output.logits.argmax(dim=-1).tolist()
                        pred_id2label = [acc.id2label[e] for e in pred]
                        true_id2label = [target_emotion]
                        acc.process(k, v, pred_id2label, true_id2label)            
        bar.update(1)
    acc.save()

def objective_eval(device, model, args, configs, vocoder, batchs, result_path):
    preprocess_config, model_config, _ = configs
    data_df = pd.read_csv(args.source, sep='|', header=None, names=['id', 'speaker_id', 'text', 'raw_text'], dtype = str)
    mcd = MCD(result_path)
    wer = WER(device, result_path)
    prompt_gpt = Prompt_GPT(device, preprocess_config)
    
    raw_text_dict, key_count = {}, 0
    for raw_text in data_df['raw_text'].unique():
        raw_text_dict[raw_text] = str(key_count)
        os.makedirs(os.path.join(result_path, str(key_count)), exist_ok=True)
        key_count += 1

    bar = tqdm(total=len(batchs))
    for batch in batchs:
        batch = batch2device(batch, device)       
        path_meta = batch[0][0].split('_')
        unmap_spker_id, wav_id, emotion = path_meta[0], '_'.join(path_meta[0:2]), path_meta[2]
        if emotion != 'Neutral':
            bar.update(1)
            continue
        for target_emotion in sorted(["Angry", "Happy", "Sad", "Surprise"]):
            # check reference target sample exists 
            target_df = data_df[data_df['id'].str.contains(target_emotion)
                    & (data_df['raw_text'] == batch['raw_text'][0])
                    & (data_df['speaker_id'] == unmap_spker_id)]
            if len(target_df) == 0:
                print(f'Conditions {target_emotion}, {batch["raw_text"][0]}, {batch["speaker_id"][0]} not found in data_df')
                continue
            # target_wav_id = '_'.join(target_df['id'].values[0].split('_')[0:2])
            target_wav_id = target_df['id'].values[0]
            if args.intensity_label == 'intensity':
                st_path = os.path.join(
                        os.path.dirname(result_path), 
                        'intensity_embeds', unmap_spker_id, f"{target_wav_id}_{target_emotion}.npy")
                if not os.path.exists(st_path):
                    print(f'{st_path} not found')
                    continue
                save_path = os.path.join(result_path, str(raw_text_dict[batch['raw_text'][0]]))
                set1_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_no_prompt.wav')
                set2_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word-utter_prompt.wav')
                set3_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word_prompt.wav')
                target_intensity_embed_cmt = np.load(st_path)
            else:
                save_path = os.path.join(result_path, str(raw_text_dict[batch['raw_text'][0]]))
                set1_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_no_prompt.wav')
                set2_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word-utter_prompt.wav')
                set3_path = os.path.join(save_path, f'{wav_id}_{target_emotion}_word_prompt.wav')
                target_intensity_embed_cmt = None

            prompt_path = os.path.join(save_path, f"{wav_id}_{target_emotion}_prompt.txt")
            prompt_output = prompt_gpt.process(batch, target_emotion, prompt_path)
            if prompt_output == None:
                continue
            utterance_df, word_df, batch = prompt_output
            mcd.process_ref(
                raw_path = preprocess_config["path"]["raw_path"],
                save_path = save_path,
                basename = target_df['id'].values[0])
            wer.process_ref(
                raw_path = preprocess_config["path"]["raw_path"],
                basename = target_df['id'].values[0])
            emt_embed_path = os.path.join(
                os.path.dirname(result_path), 
                'emotion_embeds', unmap_spker_id,  f"{target_wav_id}_{target_emotion}.npy")
            if not os.path.exists(emt_embed_path):
                print(f'{emt_embed_path} not found')
                continue
            target_emotion_embed_cmt = np.load(emt_embed_path)
            
            setting_dict = {'set1': {'path': set1_path, 'global_control': None, 'local_control': None, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': target_intensity_embed_cmt},
                            'set2': {'path': set2_path, 'global_control': utterance_df, 'local_control': word_df, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': target_intensity_embed_cmt},
                            'set3': {'path': set3_path, 'global_control': None, 'local_control': word_df, 'emotion_embed_cmt': target_emotion_embed_cmt, 'intensity_embed_cmt': target_intensity_embed_cmt}}
            
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
                            emotion_embed_cmt=v['emotion_embed_cmt'],
                            intensity_embed_cmt=v['intensity_embed_cmt'])
                        synth_samples(batch, output, vocoder, model_config, preprocess_config, v['path'], args)
                    mcd.process_synth(
                        key = k,
                        value = v,
                        save_path = save_path,
                    )
                    wer.process_synth(
                        key = k,
                        value = v
                    )
        bar.update(1)
    mcd.save()
    # wer.save()

## python objective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --checkpoint Audio_Data/output/ESD_Emo_Int_unsup_ph/ --dataset ESD --emotion_label 1 --intensity_label 0
## python objective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --checkpoint Audio_Data/output/ESD_Emo_Int_unsup_ph/ --dataset ESD --emotion_label 1 --intensity_label 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--source", type=str, default=None, help="path to a source file with format like train.txt and val.txt")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to the checkpoint")
    parser.add_argument("--emotion_label", type = int, default = 0, help = "Whether to use emotion label")
    parser.add_argument("--intensity_label", type = int, default = 0, help = "Whether to use intensity label")
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
    
    result_path = os.path.join(args.checkpoint, 'result', str(args.restore_step), 'objective_eval')
    os.makedirs(result_path, exist_ok=True)
    # Set Device
    torch.manual_seed(train_config["seed"])
    print("Device of CompTransTTS:", device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    dataset = LabeledDataset(args.source, preprocess_config, model_config)
    # model = None
    batchs = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate_fn)
    model = get_model(args, configs, device, train=False, 
                                emt_labels = args.emotion_label, 
                                int_labels = args.intensity_label)
    
    objective_eval(device, model, args, configs, vocoder, batchs, result_path)
    # eval(device, model, args, configs, vocoder, batchs, result_path)
    # acc(device, model, args, configs, batchs, result_path)
