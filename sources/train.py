import argparse
import os
import torch
import numpy as np

from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel


from tqdm import tqdm
from evaluate import evaluate
from dataset import Dataset
from model import CompTransTTSLoss
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, log, synth_one_sample, batch2device, plot_classifier

torch.backends.cudnn.benchmark = True
'''Adjust the device_id to the available GPU device id. e.g 0, 1, 2, 3'''
device_id = 0
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)
'''This script is used to train the CompTransTTS model.'''
def train(rank, args, configs, batch_size, num_gpus, device_id = None):
    preprocess_config, model_config, train_config = configs
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            init_method=train_config["dist_config"]['dist_url'],
            world_size=train_config["dist_config"]['world_size'] * num_gpus,
            rank=rank,
        )
    if device_id is not None:
        device = torch.device('cuda:{:d}'.format(device_id))

    # Get dataset
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    dataset_tag = "unsup" if learn_alignment else "sup"
    pitch_type = preprocess_config['preprocessing']['pitch']['pitch_type']
    if args.checkpoint is not None:
        args.checkpoint = os.path.join(os.path.dirname(train_config['path']), 
                                       f'{args.checkpoint}_{dataset_tag}_{pitch_type}')
    
    print("Getting Dataset ...")
    dataset = Dataset(
        "train_{}.txt".format(dataset_tag), preprocess_config, model_config, train_config, sort=True, drop_last=True, 
        with_emt = args.emotion_label, with_int = args.intensity_label
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    print("Getting Loader ...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    print("Preparing Model ...")
    model, optimizer = get_model(args, configs, device, train=True, 
                                 emt_labels = args.emotion_label, 
                                 int_labels = args.intensity_label)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Training
    if args.checkpoint is not None:
        step = 1
    else:
        step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    if rank == 0:
        print("Number of CompTransTTS Parameters: {}\n".format(get_param_num(model)))
        # Init logger
        model_path = train_config["path"]
        if args.emotion_label:
            model_path += '_Emo'
        if args.intensity_label:
            model_path += '_Int'
        model_path += f'_{dataset_tag}_{pitch_type}'

        ckpt_path = os.path.join(model_path, "ckpt")
        log_path = os.path.join(model_path, "log")
        result_path = os.path.join(model_path, "result")
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)

        train_log_path = os.path.join(log_path, "train")
        val_log_path = os.path.join(log_path, "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)
        vals_info = {'Emotion':{'steps':[], 'losses':[], 'accs':[]},
                     'Intensity':{'steps':[], 'losses':[], 'accs':[]}}
        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = args.restore_step
        outer_bar.update()

    train = True
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
            
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if train == False:
                break
            if args.emotion_label and step > model_config['emotion_encoder']['max_train_steps']:
                model.emotion_encoder.eval()
            if args.intensity_label and step > model_config['intensity_encoder']['max_train_steps']:
                model.intensity_encoder.eval()
            for batch in batchs:
                batch = batch2device(batch, device)
                with amp.autocast(args.use_amp):
                    # Forward
                    output = model(*(batch), step=step)
                    batch['pitch_data'] = output.p_targets
                    batch['energy'] = output.e_targets
                    # Cal Loss
                    losses = Loss(batch, output, step=step)
                    total_loss = losses.total_loss
                    total_loss = total_loss / grad_acc_step
                
                # Backward
                scaler.scale(total_loss).backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                lr = optimizer.step_and_update_lr(scaler)
                scaler.update()
                optimizer.zero_grad()

                if rank == 0:
                    if step % log_step == 0:
                        message1 = "Step {}/{}, ".format(step, total_step)
                        for k,v in losses.items():
                            if type(v) == dict:
                                for k1, v1 in v.items():
                                    message1 += f' {k1}: {round(v1.item(),3)}'
                            else:
                                message1 += f' {k}: {round(v.item(),3)}'

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + "\n")

                        outer_bar.write(message1)
                        log(train_logger, step, losses=losses, lr=lr)

                    if step % synth_step == 0:
                        figs, fig_attn, wav_reconstruction, wav_prediction, _ = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        if fig_attn is not None:
                            log(
                                train_logger,
                                step,
                                img=fig_attn,
                                tag="Training/attn",
                            )
                        log(
                            train_logger,
                            step,
                            figs=figs,
                            tag="Training",
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            step,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/reconstructed",
                        )
                        log(
                            train_logger,
                            step,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/synthesized",
                        )

                    if step % val_step == 0 or step == 1:
                        model.eval()
                        message, vals_info = evaluate(args, device, model, step, configs, val_logger, vocoder, losses, vals_info)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)
                        plot_classifier(vals_info['Emotion'], os.path.join(val_log_path, 'emotion_plot.png'))
                        plot_classifier(vals_info['Intensity'], os.path.join(val_log_path, 'intensity_plot.png'))
                        model.train()

                    if args.emotion_label and step > model_config['emotion_encoder']['max_train_steps']:
                        model.emotion_encoder.eval()
                    if args.intensity_label and step > model_config['intensity_encoder']['max_train_steps']:
                        model.intensity_encoder.eval()
                    
                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                ckpt_path,
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1

# python train.py --dataset LibriTTS
# python train.py --dataset ESD --restore_step 900000 --checkpoint LibriTTS --emotion_label 1 --intensity_label 1
# python train.py --dataset IMOCap --restore_step 900000 --checkpoint LibriTTS --emotion_label 1 --intensity_label 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type= int, default = 0)
    parser.add_argument("--checkpoint", type = str, default = None, help="The checkpoints folder path")
    parser.add_argument("--emotion_label", type = int, default = 0, help = "Whether to use emotion label")
    parser.add_argument("--intensity_label", type = int, default = 0, help = "Whether to use intensity label")
    parser.add_argument("--dataset", type = str, required = True, help="name of dataset")

    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    '''
    Revise train_config, various steps are adjusted according to the total_step.
    Total steps is set based on the training set size
    '''
    total_step = train_config["step"]["total_step"]
    train_config['optimizer']['warm_up_step'] = int(train_config['optimizer']['warm_up_step'] * total_step / 900000)
    train_config['optimizer']['anneal_steps'] = [int(s * total_step / 900000) for s in train_config['optimizer']['anneal_steps']]
    train_config['step']['var_start_steps'] = int(train_config['step']['var_start_steps'] * total_step / 900000)
    for k in train_config['duration'].keys():
        train_config['duration'][k] = int(train_config['duration'][k] * total_step / 900000)
    train_config['prosody']['prosody_loss_enable_steps'] = int(train_config['prosody']['prosody_loss_enable_steps'] * total_step / 900000)
    
    configs = (preprocess_config, model_config, train_config)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Set Device
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(" ---> Type of Building Block:", model_config["block_type"])
    print(" ---> Type of Duration Modeling:", "unsupervised" if model_config["duration_modeling"]["learn_alignment"] else "supervised")
    print(" ---> Type of Prosody Modeling:", model_config["prosody_modeling"]["model_type"])
    print("=================================================================================================")
    print("Prepare training ...")

    train(0, args, configs, batch_size, num_gpus=1, device_id = device_id)
