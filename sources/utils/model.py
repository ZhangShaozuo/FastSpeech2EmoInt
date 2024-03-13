import os
import torch
import numpy as np
from model import CompTransTTS, ScheduledOptim
# from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.pretrained import HIFIGAN
from transformers import SpeechT5HifiGan

def get_model(args, configs, device, train=False, emt_labels=False, int_labels=False):
    (preprocess_config, model_config, train_config) = configs

    model = CompTransTTS(preprocess_config, model_config, train_config, emt_labels, int_labels).to(device)
    if args.restore_step:
        if args.checkpoint:
            '''If checkpoint is provided, the training setting will be fine-tuning.'''
            ckpt_path = os.path.join(args.checkpoint, 'ckpt','{}.pth.tar'.format(args.restore_step))
        else:
            '''If checkpoint is not provided, the training setting will be resumed from the last step.'''
            base_path = train_config["path"]
            learn_alignment = model_config["duration_modeling"]["learn_alignment"]
            dataset_tag = "unsup" if learn_alignment else "sup"
            pitch_type = preprocess_config['preprocessing']['pitch']['pitch_type']
            if emt_labels:
                base_path += '_Emo'
            if int_labels:
                base_path += '_Int'
            ckpt_path = os.path.join(
                f'{base_path}_{dataset_tag}_{pitch_type}', 'ckpt',"{}.pth.tar".format(args.restore_step))
            print(f'Resume training from {ckpt_path}, please check the path.')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        # torch.save(model.emotion_classifier.state_dict(), './Audio_Data/output/hubert_50000.pth.tar')
        # st_ckpt = torch.load(f'{base_path}_Attn/ckpt/50000.pth.tar', map_location=device)
        # model.intensity_regressor.load_state_dict(st_ckpt["model"], strict=True)
    if train:
        if args.checkpoint:
            '''For fine-tuning, the optimizer will be initialized.'''
            scheduled_optim = ScheduledOptim(
                model, train_config, model_config, 0
            )
        else:
            '''For resuming, the optimizer will be loaded from the checkpoint.'''
            scheduled_optim = ScheduledOptim(
                model, train_config, model_config, args.restore_step
            )
            if args.restore_step:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    elif name == "SpeechBrain":
        vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz")

    if name == 'SpeechT5':
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "SpeechBrain":
            wavs = vocoder.decode_batch(mels).squeeze(1)
        elif name == "SpeechT5":
            mels = mels / np.log(10)
            wavs = vocoder(mels.transpose(1,2))

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]
    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
