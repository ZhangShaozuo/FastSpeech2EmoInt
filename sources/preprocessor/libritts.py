import os
import json
import librosa
import torchaudio
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

        
def prepare_align(config):
    '''train|val is pre-defined in the dataset release'''
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    side_out_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    speaker_idx = 0
    train_in_dir = os.path.join(in_dir, 'train-clean-100') #33236
    dev_in_dir = os.path.join(in_dir, 'dev-clean') #5736
    train_spker_set = set(os.listdir(train_in_dir))
    dev_spker_set = set(os.listdir(dev_in_dir))
    assert train_spker_set.isdisjoint(dev_spker_set) == True
    with open(os.path.join(side_out_dir, 'train_spker.json'), 'w') as f:
        json.dump(list(train_spker_set), f)
    with open(os.path.join(side_out_dir, 'dev_spker.json'), 'w') as f:
        json.dump(list(dev_spker_set), f)
    split_paths = [train_in_dir, dev_in_dir]
    progress_bar = tqdm(total=38972, desc="Align Bar {}".format(count), position=0)
    speaker_bar = tqdm(total=len(train_spker_set)+len(dev_spker_set), desc="Speaker Bar {}".format(speaker_idx), position=1)
    for split_path in split_paths:
        for _, spker in enumerate(os.listdir(split_path)):
            spker_path = os.path.join(split_path, spker)
            for _, chapter in enumerate(os.listdir(spker_path)):
                chapter_path = os.path.join(spker_path, chapter)
                for wav_path in Path(chapter_path).glob('**/*.wav'):
                    wav_path = str(wav_path)
                    wav_basename = os.path.basename(wav_path)
                    text_basename = wav_basename.replace('.wav', '.lab')
                    text_path = wav_path.replace('.wav', '.normalized.txt')          
                    wav, _ = librosa.load(wav_path, sr = sampling_rate)
                    ### this line is useless
                    wav = wav / max(abs(wav)) * max_wav_value
                    with open(text_path, 'r') as f:
                        text = f.readline().strip()
                        text = _clean_text(text, cleaners)
                    
                    out_dir_spker = os.path.join(out_dir, spker)
                    if not os.path.exists(out_dir_spker):
                        os.makedirs(out_dir_spker)
                    wavfile.write(
                        os.path.join(out_dir_spker, wav_basename),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(os.path.join(out_dir_spker, text_basename),"w") as f:
                        f.write(text)
                    count += 1
                    progress_bar.update(1)
            speaker_bar.update(1)
    print("Total number of files: ", count)
