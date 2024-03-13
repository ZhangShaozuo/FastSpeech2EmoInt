import json
import math
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from text import text_to_sequence
from transformers import Wav2Vec2FeatureExtractor
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D, IndexedDict
from utils.pitch_tools import norm_interp_f0, get_lf0_cwt
import re
from text import grapheme_to_phoneme
from g2p_en import G2p

NAME_COL = 'fileName'
EMOTION_COL = 'Emotion'
INTENSITY_COL = 'IntensityLabel'

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False, 
        with_emt=False, with_int=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocess_config = preprocess_config
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.dataset_tag = "unsup" if self.learn_alignment else "sup"
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]+f'_{self.dataset_tag}'
        
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        self.with_emt = with_emt
        self.with_int = with_int
        if with_emt or with_int:
            self.wav_path = preprocess_config["path"]["raw_path"]
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_config['emotion_encoder']['model_ckpt'])
            self.sampling_rate = self.feature_extractor.sampling_rate
        if with_emt:
            self.emotion_set = preprocess_config["emotion_labels"]
        if with_int:
            self.intensity_dit = self.preprocessed_path+'/relative_attr/labels'
            self.intensity_df = self.collate_intensity()

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        dataset_tag = self.dataset_tag
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(dataset_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(dataset_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        f0_path = os.path.join(
            self.preprocessed_path,
            "f0_{}".format(dataset_tag),
            "{}-f0-{}.npy".format(speaker, basename),
        )
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0, self.preprocess_config["preprocessing"]["pitch"])
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}_{}".format(dataset_tag, self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            duration = None
            mel2ph = None
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            mel2ph_path = os.path.join(
                self.preprocessed_path,
                "mel2ph",
                "{}-mel2ph-{}.npy".format(speaker, basename),
            )
            mel2ph = np.load(mel2ph_path)
            attn_prior = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        cwt_spec = f0_mean = f0_std = f0_ph = None
        if self.pitch_type == "cwt":
            cwt_spec_path = os.path.join(
                self.preprocessed_path,
                "cwt_spec_{}".format(dataset_tag),
                "{}-cwt_spec-{}.npy".format(speaker, basename),
            )
            cwt_spec = np.load(cwt_spec_path)
            f0cwt_mean_std_path = os.path.join(
                self.preprocessed_path,
                "f0cwt_mean_std_{}".format(dataset_tag),
                "{}-f0cwt_mean_std-{}.npy".format(speaker, basename),
            )
            f0cwt_mean_std = np.load(f0cwt_mean_std_path)
            f0_mean, f0_std = float(f0cwt_mean_std[0]), float(f0cwt_mean_std[1])

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "f0": f0,
            "uv": uv,
            "cwt_spec": cwt_spec,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "energy": energy,
            "duration": duration,
            "mel2ph": mel2ph,
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,
        }
        if self.with_emt:
            wav_path = os.path.join(
                self.preprocessed_path,
                'wav', '{}-wav-{}.npy'.format(speaker, basename))
            wav = np.load(wav_path)
            sample['wav'] = wav
            sample['emotion'] = basename.split('_')[-1]
        if self.with_int:
            if sample['emotion'] == 'Neutral':
                sample['intensity'] = 0
            else:
                sample['intensity'] = self.intensity_df[self.intensity_df[NAME_COL]==basename][INTENSITY_COL].item()
        return sample
    
    def collate_intensity(self):
        collated_df = pd.DataFrame()
        for _, speaker in enumerate(os.listdir(self.intensity_dit)):
            spker_path = os.path.join(self.intensity_dit, speaker)
            for emotion in os.listdir(spker_path):
                if not emotion.endswith('.csv'):
                    continue
                emt_path = os.path.join(spker_path, emotion)
                i_df = pd.read_csv(emt_path)
                if self.preprocess_config["dataset"] == 'ESD':
                    emotion_tag = emotion.split('.')[0]
                    i_df[NAME_COL] = i_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0]+f'_{emotion_tag}')
                else:
                    i_df[NAME_COL] = i_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0])
                i_df = i_df[~i_df['feature_paths'].str.contains('Neutral')]
                i_df.rename(columns={'X_attr': INTENSITY_COL}, inplace=True)
                collated_df = pd.concat([collated_df, i_df[[NAME_COL, INTENSITY_COL, 'bin_label']]], ignore_index=True)
        return collated_df

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        f0s = [data[idx]["f0"] for idx in idxs]
        uvs = [data[idx]["uv"] for idx in idxs]
        cwt_specs = f0_means = f0_stds = f0_phs = None
        if self.pitch_type == "cwt":
            cwt_specs = [data[idx]["cwt_spec"] for idx in idxs]
            f0_means = [data[idx]["f0_mean"] for idx in idxs]
            f0_stds = [data[idx]["f0_std"] for idx in idxs]
            cwt_specs = pad_2D(cwt_specs)
            f0_means = np.array(f0_means)
            f0_stds = np.array(f0_stds)
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        mel2phs = [data[idx]["mel2ph"] for idx in idxs] if not self.learn_alignment else None
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        f0s = pad_1D(f0s)
        uvs = pad_1D(uvs)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)
            mel2phs = pad_1D(mel2phs)
        sample = IndexedDict()
        pitch_data = {}
        sample['id'] = ids
        sample['raw_text'] = raw_texts
        sample['speaker'] = speakers
        sample['text'] = texts
        sample['text_len'] = text_lens
        sample['max_text_len'] = max(text_lens)
        sample['mel'] = mels
        sample['mel_len'] = mel_lens
        sample['max_mel_len'] = max(mel_lens)
        pitch_data['pitch'] = pitches
        pitch_data['f0'] = f0s
        pitch_data['uv'] = uvs
        pitch_data['cwt_spec'] = cwt_specs
        pitch_data['f0_mean'] = f0_means
        pitch_data['f0_std'] = f0_stds
        pitch_data['mel2ph'] = mel2phs
        sample['pitch_data'] = pitch_data
        sample['energy'] = energies
        sample['duration'] = durations
        sample['attn_prior'] = attn_priors
        sample['spker_embed'] = spker_embeds
        if self.with_emt or self.with_int:
            wavs = [data[idx]["wav"] for idx in idxs]
            wavs_features = self.feature_extractor(wavs, 
                                                sampling_rate = self.sampling_rate,
                                                padding = True,
                                                return_tensors = 'pt',
                                                return_attention_mask = True)
            wavs_iv = wavs_features.input_values
            wavs_am = wavs_features.attention_mask
            
            sample['wav'] = wavs_iv
            sample['wav_attn_mask'] = wavs_am
        if self.with_emt:
            emotions = [self.emotion_set.index(data[idx]["emotion"]) for idx in idxs]
            emotions = np.array(emotions)
            sample['emotion'] = emotions
        if self.with_int:
            intensities = [data[idx]["intensity"] for idx in idxs]
            intensities = np.array(intensities)
            sample['intensity'] = intensities
        return sample

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        return (basename, speaker_id, phone, raw_text, spker_embed)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds


class LabeledDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config, speaker_id=None):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.g2p = G2p()
        self.basename, self.speaker, self.text, self.raw_text= self.process_meta(
            filepath, speaker_id)
        dataset_tag = "unsup" if model_config["duration_modeling"]["learn_alignment"] else "sup"
        self.preprocessed_path = '{}_{}'.format(preprocess_config["path"]["preprocessed_path"], dataset_tag)
        with open(
            os.path.join(
                self.preprocessed_path, "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)
        
        self.dataset_name = preprocess_config["dataset"]
        
        
        # self.sampling_rate = self.feature_extractor.sampling_rate
        self.sampling_rate = 16000
        if self.dataset_name == 'ESD':
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                                model_config['emotion_encoder']['model_ckpt'])
            self.emotion_set = sorted(["Angry", "Happy", "Sad", "Surprise"])
            self.intensity_dit = self.preprocessed_path+'/relative_attr/labels'
            self.intensity_df = self.collate_intensity()
            self.intensity_set = sorted(self.intensity_df[INTENSITY_COL].unique().tolist())
        elif self.dataset_name == 'IMOCap':
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                                model_config['emotion_encoder']['model_ckpt'])
            self.emotion_set = sorted(['Anger', 'Excited', 'Frustration', 'Happiness', 'Neutral', 'Sadness', 'Surprise'])
            self.intensity_dit = self.preprocessed_path+'/relative_attr/labels'
            self.intensity_df = self.collate_intensity()
            self.intensity_set = sorted(self.intensity_df[INTENSITY_COL].unique().tolist())

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        sample = IndexedDict()
        sample['id'] = basename
        sample['speaker_id'] = speaker_id
        sample['phone'] = phone
        sample['raw_text'] = raw_text
        sample['spker_embed'] = spker_embed
        # wav_path = os.path.join(
        #         self.preprocessed_path,
        #         'wav', '{}-wav-{}.npy'.format(speaker, basename))
        # wav = np.load(wav_path)
        # sample['wav'] = wav
        # sample['emotion'] = basename.split('_')[-1]
        # if self.intensity_set is not None:
        #     if sample['emotion'] == 'Neutral':
        #         sample['intensity'] = 'Normal'
        #     else:
        #         basename_ = '_'.join(basename.split('_')[:-1])
        #         # retrieve the intensity from self.intensity_df, where NAME_COL == basename_
        #         sample['intensity'] = self.intensity_df[self.intensity_df[NAME_COL]==basename_][INTENSITY_COL].item()
        return sample

    def process_meta(self, filename, speaker_id=None):

        # i_f = os.path.join(filename, 'val_unsup.txt')
        # o_f = os.path.join(filename, 'val_prompt.txt')
        with open(filename, "r", encoding="utf-8") as f:
            name, speaker, text, raw_text = [], [], [], []
            for line in f.readlines():
                if len(line.strip("\n").split("|")) == 3:
                    n, s, r = line.strip("\n").split("|")
                    phone = grapheme_to_phoneme(r, self.g2p)
                    phones = "{" + "}{".join(phone) + "}"
                    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
                    t = phones.replace("}{", " ")
                    text.append(t)
                elif len(line.strip("\n").split("|")) == 4:
                    n, s, t, r= line.strip("\n").split("|")
                    text.append(t)
                if speaker_id is None:
                    speaker.append(s)
                else:
                    speaker.append(speaker_id)
                name.append(n)
                raw_text.append(r)
        # with open(o_f, "w", encoding="utf-8") as f:
        #     for i in range(len(name)):
        #         f.write('{}|{}|{}\n'.format(name[i], speaker[i], raw_text[i]))
        return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        # wavs = [d[5] for d in data]
        # emotions = [d[6] for d in data]
        # intensities = [d[7] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)
        sample = IndexedDict()
        sample['id'] = ids
        sample['raw_text'] = raw_texts
        sample['speaker_id'] = speakers
        sample['text'] = texts
        sample['text_lens'] = text_lens
        sample['max_text_len'] = max(text_lens)
        sample['spker_embed'] = spker_embeds
        # emotions = [self.emotion_set.index(emotion) for emotion in emotions]
        # emotions = np.array(emotions)
        # if self.intensity_set is not None:
        #     intensities = [self.intensity_set.index(intensity) for intensity in intensities]
        #     intensities = np.array(intensities)

        # wavs_features = self.feature_extractor(wavs, 
        #                                     sampling_rate = self.sampling_rate,
        #                                     padding = True,
        #                                     return_tensors = 'pt',
        #                                     return_attention_mask = True)
        # wavs_iv = wavs_features.input_values
        # wavs_am = wavs_features.attention_mask
        # sample['wav'] = wavs_iv
        # sample['wav_attn_mask'] = wavs_am
        # sample['emotion'] = emotions
        # sample['intensity'] = intensities

        return sample
    
    def collate_intensity(self):
        collated_df = pd.DataFrame()
        for spker_id, speaker in enumerate(os.listdir(self.intensity_dit)):
            spker_path = os.path.join(self.intensity_dit, speaker)
            for emotion in os.listdir(spker_path):
                ## if file end with .csv
                if emotion.endswith('.csv'):
                    e_path = os.path.join(spker_path, emotion)
                    i_df = pd.read_csv(e_path)
                    i_df[NAME_COL] = i_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0])
                    ### extract the rows that feature paths don't include Neutral
                    i_df = i_df[~i_df['feature_paths'].str.contains('Neutral')]
                    ## concat it with collated_df
                    i_df.rename(columns={'bin_label': INTENSITY_COL}, inplace=True)
                    collated_df = pd.concat([collated_df, i_df[[NAME_COL, INTENSITY_COL]]], ignore_index=True)
        ## map 0 to Normal, 1 to High in IntensityLabel
        collated_df.loc[collated_df[INTENSITY_COL] == 0, INTENSITY_COL] = 'Normal'
        collated_df.loc[collated_df[INTENSITY_COL] == 1, INTENSITY_COL] = 'High'
        return collated_df
    

class SER_Dataset(Dataset):
    def __init__(
        self, filename, drop_last=False
    ):  
        # self.preprocessed_path = './Audio_Data/preprocessed_data/IMOCap_unsup'
        self.preprocessed_path = './Audio_Data/preprocessed_data/ESD_unsup'
        # self.st_path = os.path.join(self.preprocessed_path, 'ESD_scores.csv')
        # self.label_df = pd.read_csv(self.st_path)
        self.intensity_dit = os.path.join(self.preprocessed_path, 'relative_attr', 'labels')
        self.label_df = self.collate_intensity()
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        self.wav_path = './Audio_Data/raw_data/ESD'
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960")
        self.sampling_rate = self.feature_extractor.sampling_rate
        
        ''' For ESD '''
        # self.emotion_set = sorted(["Angry", "Happy", "Neutral", "Sad", "Surprise"])
        self.drop_last = drop_last
        self.batch_size = 16

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]

        sample = {
            "id": basename
        }
        wav_path = os.path.join(
            self.preprocessed_path,
            'wav', '{}-wav-{}.npy'.format(speaker, basename))
        wav = np.load(wav_path)
        sample['wav'] = wav
        '''For ESD'''
        sample['emotion'] = basename.split('_')[-1]
        # breakpoint()
        if basename.split('_')[-1] == 'Neutral':
            sample['intensity'] = 0
        else:
            breakpoint()
            sample['intensity'] = self.label_df[self.label_df[NAME_COL]==basename][INTENSITY_COL].values[0]
        '''For CREMA-D'''
        # emotion = self.label_df[self.label_df[NAME_COL]==basename][EMOTION_COL].item()
        # intensity = self.label_df[self.label_df[NAME_COL]==basename][INTENSITY_COL].item()
        # if intensity == "Low":
        #     ### There are only 18 Low labels, should not affect much.
        #     intensity == "Medium"
        # sample['intensity'] = intensity
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                # if (n not in self.label_df[NAME_COL].values) and n.split('_')[-1] != 'Neutral':
                #     continue
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        sample = IndexedDict()
        sample['id'] = ids
        wavs = [data[idx]["wav"] for idx in idxs]
        '''For ESD'''
        # emotions = [self.emotion_set.index(data[idx]["emotion"]) for idx in idxs]
        emotion = [data[idx]["emotion"] for idx in idxs]
        intensity = [data[idx]["intensity"] for idx in idxs]

        wavs_features = self.feature_extractor(wavs, 
                                            sampling_rate = self.sampling_rate,
                                            padding = True,
                                            return_tensors = 'pt',
                                            return_attention_mask = True)
        wavs_iv = wavs_features.input_values
        wavs_am = wavs_features.attention_mask
        '''For ESD'''
        # emotions = np.array(emotions)
        '''For CREMA-D'''
        intensity = np.array(intensity)
        sample['wav'] = wavs_iv
        sample['wav_attn_mask'] = wavs_am
        sample['emotion'] = emotion
        sample['intensity'] = intensity
        return sample

    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    
    def collate_intensity(self):
        collated_df = pd.DataFrame()
        for _, speaker in enumerate(os.listdir(self.intensity_dit)):
            spker_path = os.path.join(self.intensity_dit, speaker)
            for emotion in os.listdir(spker_path):
                ## if file end with .csv
                if emotion.endswith('.csv'):
                    e_path = os.path.join(spker_path, emotion)
                    i_df = pd.read_csv(e_path)
                    # i_df[NAME_COL] = i_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0])
                    emotion_tag = emotion.split('.')[0]
                    i_df[NAME_COL] = i_df['feature_paths'].apply(lambda x: os.path.basename(x).split('.')[0]+f'_{emotion_tag}')
                    ### extract the rows that feature paths don't include Neutral
                    i_df = i_df[~i_df['feature_paths'].str.contains('Neutral')]
                    ## concat it with collated_df
                    i_df.rename(columns={'X_attr': INTENSITY_COL}, inplace=True)
                    collated_df = pd.concat([collated_df, i_df[[NAME_COL, INTENSITY_COL]]], ignore_index=True)
        # ## map 0 to Normal, 1 to High in IntensityLabel
        # collated_df.loc[collated_df[INTENSITY_COL] == 0, INTENSITY_COL] = 'Normal'
        # collated_df.loc[collated_df[INTENSITY_COL] == 1, INTENSITY_COL] = 'High'
        return collated_df