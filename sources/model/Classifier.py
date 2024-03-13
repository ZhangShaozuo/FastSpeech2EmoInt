import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel
)
from utils.sidekit_pooling import MeanStdPooling, AttentivePooling, AttentiveStatsPool

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.src_hidden_size, config.tgt_hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.tgt_hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x_embed = self.dense(x)
        x = self.dropout(x_embed)
        x = self.out_proj(x)
        return x, x_embed

class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)  
        self.init_weights()
        self.pool_module = AttentivePooling(num_channels=768, 
                                    num_freqs=1, 
                                    attention_channels=128, 
                                    global_context=False)
        # self.pool_module = AttentiveStatsPool(in_dim = 768, 
        #                             attention_channels=128)
        self.transpose_vec = True

        

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_base_model(self):
        for params in self.hubert.parameters():
            params.requires_grad = False
            assert params.requires_grad == False
        self.hubert.requires_grad_ = False


    def forward(self,input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if self.transpose_vec:
            hidden_states = self.pool_module(hidden_states.transpose(1,2))
        else:
            hidden_states = self.pool_module(hidden_states)
        logits, embeds = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))/100
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=embeds,
        )
