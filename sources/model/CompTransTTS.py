import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor
from utils.tools import get_mask_from_lengths

from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from .Classifier import HubertForSpeechClassification


@dataclass
class CompTransTTSOutput(ModelOutput):
    ''' Model output for CompTransTTS, easily accessible by key|index '''
    output: Optional[torch.FloatTensor] = None
    postnet_output: Optional[torch.FloatTensor] = None
    p_predictions: Optional[torch.FloatTensor] = None
    e_predictions: Optional[torch.FloatTensor] = None
    log_d_predictions: Optional[torch.FloatTensor] = None
    d_rounded: Optional[torch.FloatTensor] = None
    src_masks: Optional[torch.BoolTensor] = None
    mel_masks: Optional[torch.BoolTensor] = None
    src_lens: Optional[torch.LongTensor] = None
    mel_lens: Optional[torch.LongTensor] = None
    attn_outs: Optional[torch.FloatTensor] = None
    prosody_info: Optional[torch.FloatTensor] = None
    p_targets: Optional[torch.FloatTensor] = None
    e_targets: Optional[torch.FloatTensor] = None
    emotion_outputs: Optional[torch.FloatTensor] = None
    intensity_outputs: Optional[torch.FloatTensor] = None


def _integrate_encoder_model(model_config, labels = None):
    if labels:
        category = 'emotion'
        config = AutoConfig.from_pretrained(
                            model_config["emotion_encoder"]["model_ckpt"],
                            num_labels=len(labels),
                            label2id={label: i for i, label in enumerate(labels)},
                            id2label={i: label for i, label in enumerate(labels)},
                            finetuning_task="wav2vec2_clf"
                        )
    else:
        category = 'intensity'
        config = AutoConfig.from_pretrained(
                            model_config["emotion_encoder"]["model_ckpt"],
                            num_labels=1
                        )
    # setattr(config, 'pooling_mode', 'AttnStats')
    setattr(config, 'pooling_mode', 'Attn')
    setattr(config, 'src_hidden_size', 768*2)
    setattr(config, 'tgt_hidden_size', model_config["transformer"]["encoder_hidden"])
    hubert = HubertForSpeechClassification.from_pretrained(
            model_config[f"{category}_encoder"]["model_ckpt"],
            config=config)
    hubert.freeze_feature_extractor()
    return hubert


class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config, emt_labels=None, int_labels=None):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer_fs2":
            from .transformers.transformer_fs2 import TextEncoder, Decoder
        elif model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        elif model_config["block_type"] == "lstransformer":
            from .transformers.lstransformer import TextEncoder, Decoder
        elif model_config["block_type"] == "fastformer":
            from .transformers.fastformer import TextEncoder, Decoder
        elif model_config["block_type"] == "conformer":
            from .transformers.conformer import TextEncoder, Decoder
        elif model_config["block_type"] == "reformer":
            from .transformers.reformer import TextEncoder, Decoder
        else:
            raise NotImplementedError

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config, self.encoder.d_model)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            self.decoder.d_model,
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    self.encoder.d_model,
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    self.encoder.d_model,
                )
        if emt_labels:
            self.emotion_encoder = _integrate_encoder_model(model_config, preprocess_config['emotion_labels'])
        if int_labels:
            self.intensity_encoder = _integrate_encoder_model(model_config)

    def forward(
        self,
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        wavs=None,
        wav_attn_masks=None,
        emotions=None,
        intensities=None,
        global_control=None,
        local_control=None,
        emotion_embed_cmt=None,
        intensity_embed_cmt=None,
        step=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)

        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]
        emotion_embeds, output_e = None, None
        if emotions is not None:
            output_e = self.emotion_encoder(wavs, attention_mask = wav_attn_masks, labels = emotions)
            emotion_embeds = output_e.hidden_states
        intensity_embeds, output_i = None, None
        if intensities is not None:
            output_i = self.intensity_encoder(wavs, attention_mask = wav_attn_masks, labels = intensities)
            intensity_embeds = output_i.hidden_states        
        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
            prosody_info,
        ) = self.variance_adaptor(
            speaker_embeds,
            emotion_embeds,
            intensity_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            global_control,
            local_control,
            emotion_embed_cmt,
            intensity_embed_cmt,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output
        return CompTransTTSOutput(
            output=output,
            postnet_output=postnet_output,
            p_predictions=p_predictions,
            e_predictions=e_predictions,
            log_d_predictions=log_d_predictions,
            d_rounded=d_rounded,
            src_masks=src_masks,
            mel_masks=mel_masks,
            src_lens=src_lens,
            mel_lens=mel_lens,
            attn_outs=attn_outs,
            prosody_info=prosody_info,
            p_targets=p_targets,
            e_targets=e_targets,
            emotion_outputs=output_e,
            intensity_outputs=output_i,
        )
