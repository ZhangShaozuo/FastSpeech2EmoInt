import os
import re
import random
import librosa
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from tqdm import tqdm

from text import _clean_text

def extract_labels(emotion_dict):
    for k, v in emotion_dict.items():
        ## check if all elements in value are unique
        if len(set(v)) == len(v):
            if 'Neutral state' in v:
                v.remove('Neutral state')
            ## random pick a value from v
            emotion_dict[k] = random.choice(v)
        else:
            emotion_dict[k] = max(set(emotion_dict[k]), key = emotion_dict[k].count)
    return emotion_dict
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    count, duration_total = 0, 0
    num_annotators = 6
    pattern = re.compile(r'\[.*?\]')
    for ses_id, session in enumerate(tqdm(os.listdir(in_dir))):
        if session == 'Documentation':
            continue
        session_path = os.path.join(in_dir, session)
        for coll_id, collection in enumerate(os.listdir(session_path)):
            if collection != 'wav':
                continue
            coll_path = os.path.join(session_path, collection)
            for sub_ses_id, sub_session in enumerate(os.listdir(coll_path)):
                sub_session_path = os.path.join(coll_path, sub_session)
                ## Read transcript
                transcript_path = sub_session_path.replace('/wav/', '/transcriptions/')+'.txt'
                with open(transcript_path, "r") as f:
                    lines = f.readlines()
                transcr_dict = {}
                for line in lines:
                    key, value = line.strip().split(' [')[0], line.strip().split(']: ')[-1]
                    value = re.sub(pattern, '', value).strip()
                    transcr_dict[key] = value
                ## Read emotion
                emotion_dit = sub_session_path.replace('/wav/', '/Categorical/')
                emotion_dict = {}
                for annot_id in range(num_annotators):
                    emotion_path = emotion_dit + f'_e{annot_id+1}_cat.txt'
                    if not os.path.exists(emotion_path):
                        # print(f'Emotion file {emotion_path} does not exist.')
                        continue
                    with open(emotion_path, "r") as f:
                        lines = f.readlines()
                    for line in lines:
                        key = line.strip().split(':')[0].strip()
                        value = line.strip().split(':')[-1].split(';')[0]
                        if key not in emotion_dict.keys():
                            emotion_dict[key] = [value]
                        else:
                            emotion_dict[key].append(value)
                emotion_dict = extract_labels(emotion_dict)
                num = sum(1 for x in Path(sub_session_path).glob('**/*.wav'))
                inner_bar = tqdm(total=num, desc=f'Processing {sub_session_path.split("/")[-1]}')
                for wav_path in Path(sub_session_path).glob('**/*.wav'):
                    duration = librosa.get_duration(filename=wav_path, sr = sampling_rate)
                    if duration < 2:
                        inner_bar.update(1)
                        continue                    
                    basename = os.path.basename(wav_path).split('.')[0]
                    emotion = emotion_dict[basename]
                    ## if any of 'Other', 'Fear', 'Disgust' in emotion
                    if emotion in ['Other', 'Fear', 'Disgust', 'Others']:
                        inner_bar.update(1)
                        continue

                    if emotion == 'Neutral state':
                        emotion = 'Neutral'
                    duration_total += duration
                    count += 1
                    speaker = basename.split('_')[0]
                    
                    save_path = os.path.join(out_dir, speaker)
                    if not os.path.exists(save_path):
                        os.makedirs(os.path.join(out_dir, speaker))
                    
                    ### save wav
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(save_path, f'{basename}_{emotion}.wav'),
                        sampling_rate,
                        wav.astype(np.int16),)
                    ### save lab
                    text = _clean_text(transcr_dict[basename], cleaners)
                    if '[' in text:
                        breakpoint()

                    with open(os.path.join(out_dir, speaker, "{}_{}.lab".format(basename, emotion)), "w") as f:
                        f.write(text)
                    inner_bar.update(1)
    print(f'Average duration: {duration_total/count}')