import torch
import numpy as np
from transformers import get_scheduler
class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self.tts_params, self.enc_params = self.split_params(model)
        self._optimizer = torch.optim.Adam(
            self.tts_params,
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

        if len(self.enc_params) > 0:
            self._optimizer_enc = torch.optim.AdamW(
                self.enc_params,
                lr = 1e-6,
                betas=train_config["optimizer"]["betas"],
                eps=train_config["optimizer"]["eps"],
                weight_decay=train_config["optimizer"]["weight_decay"],
            )
            ''' for simplicity, we use the same scheduler for emotion encoder and intensity encoder '''
            print('Using the same LR scheduler for emotion encoder and intensity encoder.(set as emotion encoder)')
            self.lr_scheduler_enc = get_scheduler(
                'linear',
                self._optimizer_enc,
                num_warmup_steps=train_config["optimizer"]["warm_up_step"],
                num_training_steps=model_config["emotion_encoder"]["max_train_steps"], 
            )

    def split_params(self, model):
        tts_params, enc_params = [], []
        for name, param in model.named_parameters():
            if "emotion_encoder" in name:
                enc_params.append(param)
            elif "intensity_encoder" in name:
                enc_params.append(param)
            else:
                tts_params.append(param)
        return tts_params, enc_params
                
    def step_and_update_lr(self, scaler):
        lr = self._update_learning_rate()
        scaler.step(self._optimizer)
        if len(self.enc_params) > 0:
            scaler.step(self._optimizer_enc)
            self.lr_scheduler_enc.step()
        return lr

    def zero_grad(self):
        self._optimizer.zero_grad()
        if len(self.enc_params) > 0:
            self._optimizer_enc.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)
        if len(self.enc_params) > 0:
            self._optimizer_enc.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr
