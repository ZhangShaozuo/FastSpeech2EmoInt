import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for spker_id, speaker in enumerate(tqdm(os.listdir(in_dir))):
        spker_path = os.path.join(in_dir, speaker)
        print(f'Speaker path: {spker_path}')
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        for e_id, emotion in enumerate(os.listdir(spker_path)):
            e_path = os.path.join(spker_path, emotion)

            with open(os.path.join(spker_path, f'{speaker}.txt'), "r") as f:
                lines = f.readlines()
            texts = {}
            for line in lines:
                line = line.strip().split('\t')
                texts[line[0]] = line[1:]
            if os.path.isdir(e_path):
                for w, wav_name in enumerate(os.listdir(e_path)):
                    # if not wav_name.endswith('.wav'):
                    #     ### remove this file
                    #     os.system(f'rm {os.path.join(e_path, wav_name)}')

                    '''remove all files in the directory'''
                    if os.path.isdir(os.path.join(e_path, wav_name)):
                        ### remove this dir
                        os.system(f'rm -r {os.path.join(e_path, wav_name)}')
                    '''re-generate the wav file'''
                    # if wav_name.endswith('.wav'):
                    #     wav_path = os.path.join(e_path, wav_name)
                    #     wav, _ = librosa.load(wav_path, sampling_rate)
                    #     wav = wav / max(abs(wav)) * max_wav_value

                    #     name = wav_name.split('.')[0]
                    #     text = _clean_text(texts[name][0], cleaners)
                    #     emotion = texts[name][1]
                    #     with open(os.path.join(out_dir, speaker, "{}_{}.lab".format(name, emotion)), "w") as f:
                    #         f.write(text)
                    #     wavfile.write(
                    #         os.path.join(out_dir, speaker, "{}_{}.wav".format(name,emotion)),
                    #         sampling_rate,
                    #         wav.astype(np.int16),
                    #     )

