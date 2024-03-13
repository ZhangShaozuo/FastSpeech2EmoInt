import stat
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from utils.tools import log, synth_one_sample, batch2device
from model import CompTransTTSLoss
from model.loss import CompTransTTSLossOutput
from dataset import Dataset

def get_INT_Level(eval_threshold, emt_true, int_list):
    assert len(emt_true) == len(int_list)
    pred2lvl = []
    for idx in range(len(emt_true)):
        if int_list[idx] > eval_threshold[emt_true[idx]]:
            pred2lvl.append('High')
        else:
            pred2lvl.append('Medium')
    return pred2lvl

def evaluate(args, device, model, step, configs, logger=None, vocoder=None, losses=None, vals_info=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    dataset_tag = "unsup" if learn_alignment else "sup"
    dataset = Dataset(
        "val_{}.txt".format(dataset_tag), preprocess_config, model_config, train_config, sort = False, drop_last = False, 
        with_emt = args.emotion_label, with_int = args.intensity_label
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device).eval()
    loss_sums = CompTransTTSLossOutput(**{k:0 for k in losses.keys()})
    count = 0
    
    eval_bar = tqdm(total=len(loader), desc="Evaluate Step {}".format(step), position=1)

    stat_dict = {'emotion':{}, 'intensity':{}}
    if args.emotion_label:
        stat_dict['emotion'] = {'preds':[], 'trues':[]}
        # e_preds, e_trues = [], []
        id2label = model.emotion_encoder.config.id2label
    if args.intensity_label:
        # i_preds, i_trues = [], []
        '''Threshold are calulated from the collated intensity scores'''
        if preprocess_config['dataset'] == 'ESD':
            st_lvl_threshold = {'Angry': 65.1, 'Happy': 77.5, 'Sad': 64.7, 'Surprise': 77.8}
        elif preprocess_config['dataset'] == 'IMOCap':
            st_lvl_threshold = {'Anger': 65.1, 'Happiness': 59.3, 'Sadness': 59 , 'Excited': 61.3}
        stat_dict['intensity'] = {'preds':[], 'trues':[]}
    
    for batchs in loader:
        for batch in batchs:
            batch = batch2device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch), step=step)
                batch['pitch_data'] = output.p_targets
                batch['energy'] = output.e_targets
                # Cal Loss
                losses = Loss(batch, output, step=step)
                if args.emotion_label:
                    pred = output.emotion_outputs.logits.argmax(dim=-1).tolist()
                    pred_id2label = [id2label[e] for e in pred]
                    stat_dict['emotion']['preds'].extend(pred_id2label)
                    # e_preds.extend(e_pred_id2label)
                    emt_true = [id2label[e] for e in batch['emotion'].tolist()]
                    stat_dict['emotion']['trues'].extend(emt_true)
                    # trues.extend(e_true)

                if args.intensity_label:
                    pred = output.intensity_outputs.logits.squeeze(1).tolist()
                    int_true = batch['intensity'].tolist()
                    stat_dict['intensity']['preds'].extend(get_INT_Level(st_lvl_threshold, emt_true, pred))
                    stat_dict['intensity']['trues'].extend(get_INT_Level(st_lvl_threshold, emt_true, int_true))
                for k,v in losses.items():
                    if type(v) == dict:
                        loss_sums[k] = {k_:0 for k_ in v.keys()}
                        for k_, v_ in v.items():
                            loss_sums[k][k_] += v_.item() * len(batch[0])
                    else:
                        loss_sums[k] += losses[k].item() * len(batch[0])                
                
            eval_bar.update(1)
            count += 1
    message = f'\nValidation Step: {step}'
    for k in loss_sums.keys():
        if type(loss_sums[k]) == dict:
            message += f', {k.capitalize()}:'
            for k_ in loss_sums[k].keys():
                loss_sums[k][k_] /= len(dataset)
                message += f' {k_}: {round(loss_sums[k][k_],3)}'
        else:
            loss_sums[k] /= len(dataset)
            message += f', {k.capitalize()}: {round(loss_sums[k],3)}'
    
    if args.emotion_label:
        cr = classification_report(stat_dict['emotion']['trues'], stat_dict['emotion']['preds'], 
                                   target_names=id2label.values(), zero_division=0)
        message += f'\n{cr}'
        length = len(stat_dict['emotion']['preds'])
        acc = sum([1 for i in range(length) if stat_dict['emotion']['preds'][i] == stat_dict['emotion']['trues'][i]]) / length
        vals_info['Emotion']['steps'].append(step)
        vals_info['Emotion']['losses'].append(loss_sums['emotion_loss'])
        vals_info['Emotion']['accs'].append(acc)

    if args.intensity_label:
        cr = classification_report(stat_dict['intensity']['trues'], stat_dict['intensity']['preds'], 
                                     target_names=['Medium', 'High'], zero_division=0)
        message += f'\n{cr}'
        length = len(stat_dict['intensity']['preds'])
        acc = sum([1 for i in range(length) if stat_dict['intensity']['preds'][i] == stat_dict['intensity']['trues'][i]]) / length
        vals_info['Intensity']['steps'].append(step)
        vals_info['Intensity']['losses'].append(loss_sums['intensity_loss'])
        vals_info['Intensity']['accs'].append(acc)

    if logger is not None:
        figs, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_sums)
        if fig_attn is not None:
            log(
                logger,
                step,
                img=fig_attn,
                tag="Validation/attn",
            )
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )

    return message, vals_info