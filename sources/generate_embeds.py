import os
import torch
import argparse
import numpy as np

from tqdm.auto import tqdm

from dataset import Dataset
from utils.model import get_model
from transformers import AutoConfig
from torch.utils.data import DataLoader
from model.Classifier import HubertForSpeechClassification
from utils.tools import get_configs_of, batch2device

device_id = 0
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)
# NAME_COL = 'fileName'
# EMOTION_COL = 'Emotion'
# INTENSITY_COL = 'IntensityLabel'
# eval_thold = {'Angry': 65.1, 'Happy': 77.5, 'Sad': 64.7, 'Surprise': 77.8}

def generate_embeds(device, model, args, loader, result_path):
    # preprocess_config, model_config, train_config = configs
    bar = tqdm(total=len(loader))
    for batchs in loader:
        for batch in batchs:
            batch = batch2device(batch, device)
            path_meta = batch[0][0].split('_')
            unmap_speaker_arr = path_meta[0]
            speaker_path = os.path.join(result_path, unmap_speaker_arr)
            os.makedirs(speaker_path, exist_ok=True)
            breakpoint()
            with torch.no_grad():
                output = model(
                    *(batch),
                    global_control=None,
                    local_control=None,
                    step = args.restore_step
                )
                if args.label_type == 'emotion':
                    embeds = output.emotion_outputs.hidden_states.cpu().numpy()
                elif args.label_type == 'intensity':
                    embeds = output.intensity_outputs.hidden_states.cpu().numpy()
                for i in range(len(batch[0])):
                    np.save(
                        os.path.join(speaker_path, f'{batch[0][i]}.npy'), 
                        embeds[i])
            bar.update(1)

## python generate_embeds.py --checkpoint ESD --restore_step 450000 --label_type emotion
## python generate_embeds.py --checkpoint ESD --restore_step 450000 --label_type intensity
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="name of dataset")
    parser.add_argument("--label_type", type=str, required=True, default='emotion')
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.checkpoint)
    configs = (preprocess_config, model_config, train_config)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
        
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    dataset_tag = "unsup" if learn_alignment else "sup"
    pitch_type = preprocess_config['preprocessing']['pitch']['pitch_type']
    
    
    with_emt, with_int = True, True
    args.checkpoint = os.path.join(os.path.dirname(train_config['path']), 
                                       f'{args.checkpoint}_Emo_Int_{dataset_tag}_{pitch_type}')
    result_path = os.path.join(args.checkpoint, 'result', str(args.restore_step), f'{args.label_type}_embeds_val')
    os.makedirs(result_path, exist_ok=True)
    print("Result path: ", result_path)

    # Set Device
    torch.manual_seed(train_config["seed"])
    print("Device of CompTransTTS:", device)
    batch_size = train_config["optimizer"]["batch_size"]
    for split in ['train', 'val']:
    # for split in ['val']:
        print("Synethesizing {} set".format(split))
        dataset = Dataset(
            "{}_{}.txt".format(split, dataset_tag), preprocess_config, model_config, train_config, sort=False, drop_last=False, 
            with_emt=with_emt, with_int=with_int)
        model = get_model(args, configs, device, train=False, 
                        emt_labels=with_emt, int_labels=with_int)
        print("Preparing Model ...")
        # config = AutoConfig.from_pretrained(
        #                 model_config["emotion_classifier"]["model_ckpt"],
        #                 num_labels=1,
        #             )
        # setattr(config, 'pooling_mode', 'Attn')
        # setattr(config, 'src_hidden_size', 768*2)
        # setattr(config, 'tgt_hidden_size', model_config["transformer"]["encoder_hidden"])
        # if args.label_type == 'emotion':
        #     labels = preprocess_config['emotion_labels']
        #     config = AutoConfig.from_pretrained(
        #                         model_config["emotion_encoder"]["model_ckpt"],
        #                         num_labels=len(labels),
        #                         label2id={label: i for i, label in enumerate(labels)},
        #                         id2label={i: label for i, label in enumerate(labels)},
        #                         finetuning_task="wav2vec2_clf"
        #                     )
        # else:
        #     config = AutoConfig.from_pretrained(
        #                         model_config["emotion_encoder"]["model_ckpt"],
        #                         num_labels=1
        #                     )
        # setattr(config, 'pooling_mode', 'Attn')
        # setattr(config, 'src_hidden_size', 768*2)
        # setattr(config, 'tgt_hidden_size', model_config["transformer"]["encoder_hidden"])
        # model = HubertForSpeechClassification.from_pretrained(
        #         model_config["emotion_classifier"]["model_ckpt"],
        #         config=config).to(device)
        # ckpt = torch.load(os.path.join(args.checkpoint, 'ckpt', f'{args.restore_step}.pth.tar'), map_location=device)
        # model.load_state_dict(ckpt["model"], strict=False)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        generate_embeds(device, model, args, loader, result_path)

