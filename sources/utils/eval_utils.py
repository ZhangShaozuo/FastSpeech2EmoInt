import os
import re
import json
import math
import jiwer
import torch
from openai import OpenAI
import pysptk
import string
import librosa
import pyworld
import whisper
import numpy as np
import pandas as pd

from g2p_en import G2p
from string import punctuation
from text import text_to_sequence
from sklearn.metrics import classification_report

def generate_val_OBJ(train_txt, val_txt, out_txt):
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

class ACC():
    def __init__(self, id2label, result_path) -> None:
        self.id2label = id2label
        self.acc_scores = {}
        # self.num_samples = 0
        self.result_path = result_path
        self.f = open(os.path.join(result_path, 'acc.txt'), 'w')
        self.f.write('key|path|pred|true|is_equal\n')
    
    def process(self, key, value, p, y):
        if key not in self.acc_scores.keys():
            self.acc_scores[key] = {'pred': p, 'true': y}
        else:
            self.acc_scores[key]['pred'].extend(p)
            self.acc_scores[key]['true'].extend(y)
        # self.num_samples += len(p)
        self.f.write('{}|{}|{}|{}|{}\n'.format(key, value['path'], p, y, p==y))

    def save(self):
        self.f.close()
        save_dict = {}
        for k, v in self.acc_scores.items():
            save_dict[k] = {'acc': np.sum(np.array(v['pred'])==np.array(v['true'])) / np.array(v['pred']).shape[0]}
        json.dump(save_dict, open(os.path.join(self.result_path, 'acc.json'), 'w'), indent=4)

class MCD():
    def __init__(self, result_path) -> None:
        '''Input parameter'''
        self.FRAME_PERIOD = 5.0
        self.alpha = 0.65  # commonly used at 22050 Hz
        self.fft_size = 1024
        self.mcep_size = 25
        self.sampling_rate = 16000
        '''Meta record'''
        self.cur_ref_vec = None
        self.mcd_scores = {}
        self.num_frames = 0
        '''Output'''
        self.result_path = result_path
        self.f = open(os.path.join(result_path, 'mcd.txt'), 'w')
        
    def load_wav(self, wav_file, sr):
        wav, _ = librosa.load(wav_file, sr=sr, mono=True)
        return wav
    
    def log_spec_dB_dist(self, x, y):
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))
    
    def wav2mcep_numpy(self, wavfile, targetfile):
        # make relevant directories
        loaded_wav = self.load_wav(wavfile, sr=self.sampling_rate)
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.sampling_rate,
                                    frame_period=self.FRAME_PERIOD, fft_size=self.fft_size)

        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=self.mcep_size, alpha=self.alpha, maxiter=0,
                            etype=1, eps=1.0E-8, min_det=0.0, itype=3)
        np.save(targetfile,mgc,allow_pickle=False)

    def process_ref(self, raw_path, save_path, basename):
        unmap_spker_id = basename.split('_')[0]
        ref_path = os.path.join(raw_path, 
                            unmap_spker_id,
                            basename+'.wav')
        mcep_ref_path = os.path.join(save_path, 'mcep', 'GT_{}.npy'.format(basename))
        if not os.path.exists(mcep_ref_path):
            os.makedirs(os.path.dirname(mcep_ref_path), exist_ok=True)
            self.wav2mcep_numpy(ref_path, mcep_ref_path)
        self.cur_ref_vec = np.load(mcep_ref_path)
        self.num_frames += len(self.cur_ref_vec)
    
    def process_synth(self, key, value, save_path):
        synth_path = os.path.join(
            save_path, 'mcep', 
            os.path.basename(value['path']).replace('.wav', '.npy'))
        mcep_path = os.path.join(save_path, 'mcep', os.path.basename(value['path']).replace('.wav', '.npy'))
        if not os.path.exists(mcep_path):
            os.makedirs(os.path.dirname(mcep_path), exist_ok=True)
            self.wav2mcep_numpy(value['path'], mcep_path)
        synth_vec = np.load(synth_path)
        min_cost, _ = librosa.sequence.dtw(self.cur_ref_vec[:, 1:].T, synth_vec[:, 1:].T, 
                metric=self.log_spec_dB_dist)
        if key not in self.mcd_scores.keys():
            self.mcd_scores[key] = np.mean(min_cost)
        else:
            self.mcd_scores[key] += np.mean(min_cost)
        self.f.write('{}|{}|{}\n'.format(key, value['path'], np.mean(min_cost)/len(self.cur_ref_vec)))
    
    def save(self):
        self.f.close()
        save_dict = {}
        for k, v in self.mcd_scores.items():
            save_dict[k] = v / self.num_frames
        json.dump(save_dict, open(os.path.join(self.result_path, 'mcd.json'), 'w'), indent=4)

class WER():
    def __init__(self, device, result_path) -> None:
        # self.model = whisper.load_model("large", device=device)
        # breakpoint()
        self.model = whisper.load_model("small")
        '''Meta record'''
        self.wer_scores = {}
        self.cur_ref_text = None
        '''Output'''
        self.wer_scores = {}
        self.num_samples = 0
        self.result_path = result_path
        self.f = open(os.path.join(result_path, 'wer.txt'), 'w')
        
    
    def process_ref(self, raw_path, basename):
        unmap_spker_id = basename.split('_')[0]
        ref_path = os.path.join(raw_path, 
                            unmap_spker_id,
                            basename+'.lab')
        self.cur_ref_text = open(ref_path, 'r').readlines()[0].strip()
    
    def process_synth(self, key, value):
        result = self.model.transcribe(value['path'])['text'].strip()
        wer = jiwer.wer(self.cur_ref_text, result)
        cer = jiwer.cer(self.cur_ref_text, result)
        if key not in self.wer_scores.keys():
            self.wer_scores[key] = {'wer': wer, 'cer': cer}
        else:
            self.wer_scores[key]['wer'] += wer
            self.wer_scores[key]['cer'] += cer
        self.f.write('{}|{}|{}|{}\n'.format(key, value['path'], wer, cer))
        self.num_samples += 1

    def save(self):
        self.f.close()
        save_dict = {}
        for k, v in self.wer_scores.items():
            save_dict[k] = {'wer': v['wer'] / self.num_samples, 'cer': v['cer'] / self.num_samples}
        json.dump(save_dict, open(os.path.join(self.result_path, 'jiwer.json'), 'w'), indent=4)

class Prompt_GPT():
    def __init__(self, device, preprocess_config):
        # openai.api_type = "azure"
        # openai.api_base = "https://research2.openai.azure.com/"
        # openai.api_version = "2023-03-15-preview"
        # openai.api_key = "dc66a8a405a245c3a2325fa3bee682b9"
        self.client = OpenAI(api_key="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")
        self.prompt = open("prompt.txt").read()
        self.device = device
        self.preprocess_config = preprocess_config

    def process(self, batch, target_emotion, prompt_path):
        if os.path.exists(prompt_path):
            print(f"{prompt_path} exists")
        else:
            print(f"{prompt_path} does not exist, generating...")
            prompt_completed = self.prompt + "\nYour task: \n Target text: " + batch['raw_text'][0] + \
                                    "\nTarget emotion: " + target_emotion
            
            prompt_output = self.get_chatGPT(prompt_completed)
            if prompt_output is None:
                print("OpenAI API thinks it is harmful, continue to the next query.")
            with open(prompt_path, "w") as f:
                f.write(prompt_output)  
                    
        with open(prompt_path, "r") as f:
            prompt_output = f.readlines()
            
        # if 'Your tasks:' in prompt_output[0]:
        #     return 
        print(f'Reading prompt output from {prompt_path}')
        utterance_df, word_df, phone_seq = self.read_factors(prompt_output)
        batch['text'] = torch.from_numpy(phone_seq).long().unsqueeze(0).to(self.device)
        batch['text_lens'] = torch.Tensor([batch['text'].shape[1]]).long().to(self.device)
        batch['max_text_len'] = batch['text'].shape[1]
        assert len(phone_seq) == word_df['phone_len'].sum()
        return utterance_df, word_df, batch
    
    def get_chatGPT(self, prompt):
        # try:
        #     messages = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo", 
        #         engine="SunqiRE",
        #         messages=[{"role": "user", "content": prompt}])
        #     return messages.choices[0].message["content"]
        # except openai.error.InvalidRequestError:
        #     return None
        message = self.client.chat.completions.create(
            model = "gpt-4-1106-preview",
            messages = [{"role": "user", "content": prompt}]
        )
        return message.choices[0].message.content
    
    def read_factors(self, prompt_output):
        utterance_df, word_df = None, None
        current_df = None
        for idx, line in enumerate(prompt_output):
            line = line.strip().lower()
            if line.startswith("|"):
                values = [value.strip() for value in line.split("|") if value.strip()]
                # print(values)
                if len(values) == 3:
                    current_df = "utterance"
                elif len(values) == 4:
                    current_df = "word"
                if "---" in "".join(values):
                    continue
                if current_df == "utterance" and utterance_df is None:
                    utterance_df = pd.DataFrame(columns=values)
                elif current_df == "word" and word_df is None:
                    word_df = pd.DataFrame(columns=values)
                else:
                    # Append data to the respective dataframe
                    if current_df == "utterance":
                        utterance_df.loc[len(utterance_df)] = values
                    elif current_df == "word":
                        if len(values) != 4:
                            print('Num of columns mismatch in line: ', idx, line)
                            continue
                        elif values[0] in string.punctuation:
                            continue
                        ### control durations within (-2,2)
                        values[-1] = str(max(-2, min(2, int(values[-1]))))
                        word_df.loc[len(word_df)] = values
                        
        phone_sequence, word_df = preprocess_english(word_df, self.preprocess_config)
        return utterance_df, word_df, phone_sequence

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(word_df, preprocess_config, words=None):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = ''
    words = word_df['word'].tolist()
    words = [''.join([c for c in w if c not in punctuation]) for w in words]
    phone_lens = []
    for w in words:
        if w.lower() in lexicon:
            word_phones = lexicon[w.lower()]
        else:
            word_phones = list(filter(lambda p: p != " ", g2p(w)))
        word_phones = "{" + " ".join(word_phones) + "}"
        word_phones = re.sub(r"\{[^\w\s]?\}", "{sp}", word_phones)
        phone_lens.append(len(word_phones.split(' ')))
        phones += word_phones
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(words))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    word_df['phone_len'] = phone_lens
    return np.array(sequence), word_df