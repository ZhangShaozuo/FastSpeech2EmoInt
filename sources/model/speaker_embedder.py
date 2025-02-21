import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from deepspeaker import embedding
from pathlib import Path
from encoder import inference as encoder

class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.embedder_type = config["preprocessing"]["speaker_embedder"]
        self.embedder_path = config["preprocessing"]["speaker_embedder_path"]
        self.embedder_cuda = config["preprocessing"]["speaker_embedder_cuda"]
        self.embedder = self._get_speaker_embedder()

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == "GE2E":
            encoder.load_model(Path(self.embedder_path))
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == "GE2E":
            spker_embed = encoder.embed_utterance(audio).reshape(1, -1)
        return spker_embed
