import os
import json
import scipy
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from rank_svm import *
from sklearn.mixture import GaussianMixture as GMM
# from scipy.io import savemat
from tqdm.auto import tqdm
def load_data(args):
    datadict = np.load(datadict_path, allow_pickle=True).item()
    datadict['feat'] = datadict['feat'][datadict['speakers'] == args.speaker].astype(np.float64)
    datadict['feature_paths'] = np.array(datadict['feature_paths'])[datadict['speakers'] == args.speaker]
    datadict['class_labels'] = datadict['class_labels'][datadict['speakers'] == args.speaker]
    if args.emotion in emotion_dict.keys():
        emo_list = [emotion_dict['Neutral'], emotion_dict[args.emotion]] # emo_list = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        datadict['feat'] = datadict['feat'][np.isin(datadict['class_labels'], emo_list)].astype(np.float64)
        datadict['feature_paths'] = datadict['feature_paths'][np.isin(datadict['class_labels'], emo_list)]
        datadict['class_labels'] = datadict['class_labels'][np.isin(datadict['class_labels'], emo_list)]
    return datadict

def create_O_S_mat(datadict):
    S_row, S_column, S_value, S_cnt = [], [], [], 0
    O_row, O_column, O_value, O_cnt = [], [], [], 0
    X = datadict['feat']
    progress_bar = tqdm(range(len(datadict['class_labels'])))
    for i, ad1_lab in enumerate(datadict['class_labels']):

        for j, ad2_lab in enumerate(datadict['class_labels'][i+1:]):

            if cat_ordering[ad1_lab] == cat_ordering[ad2_lab]:
                
                # print i, ad1_lab, j, ad2_lab
                S_row.append(S_cnt)
                S_column.append(i)
                S_value.append(-1)
                S_row.append(S_cnt)
                S_column.append(i + j + 1)
                S_value.append(1)
                S_cnt += 1

                S_row.append(S_cnt)
                S_column.append(i)
                S_value.append(1)
                S_row.append(S_cnt)
                S_column.append(i + j + 1)
                S_value.append(-1)
                S_cnt += 1

            elif cat_ordering[ad1_lab] < cat_ordering[ad2_lab]:
                O_row.append(O_cnt)
                O_column.append(i)
                O_value.append(-1)
                O_row.append(O_cnt)
                O_column.append(i + j + 1)
                O_value.append(1)
                O_cnt += 1

            elif cat_ordering[ad1_lab] > cat_ordering[ad2_lab]:
                O_row.append(O_cnt)
                O_column.append(i)
                O_value.append(1)
                O_row.append(O_cnt)
                O_column.append(i + j + 1)
                O_value.append(-1)
                O_cnt += 1
        progress_bar.update(1)

    S = csr_matrix((S_value, (S_row, S_column)),(S_cnt, datadict['feat'].shape[0]))
    O = csr_matrix((O_value, (O_row, O_column)),(O_cnt, datadict['feat'].shape[0]))
    C_O = scipy.matrix(0.1 * np.ones([O_cnt, 1]))
    C_S = scipy.matrix(0.1 * np.ones([S_cnt, 1]))
    X = scipy.matrix(X)
    # savemat("X.mat",{'X':X})
    # savemat("S.mat",{'S':S})
    # savemat("O.mat",{'O':O})
    # savemat("C_O.mat",{'C_O':C_O})
    # savemat("C_S.mat",{'C_S':C_S})
    return X, S, O, C_S, C_O
    

def eval(args, datadict):
    attr_weights = []
    w = np.load(os.path.join(weights_dit, args.speaker, f'weights_{args.emotion}.npy'))
    attr_weights.append(w.T.tolist()[0])
    attr_weights = np.matrix(attr_weights)
    X = np.matrix(datadict['feat'])
    num_attr = 1 # only emotion intensity
    num_im = datadict['feat'].shape[0]
    X_attr = np.zeros((num_im, num_attr))
    ### find the index of 0 in cat_ordering

    neutral_idx = np.where(cat_ordering == 0)[0][0]
    if args.emotion in emotion_dict.keys():
        seen = [neutral_idx, emotion_dict[args.emotion]]
    else:
        seen = emotion_dict.values()
    unseen = []
    for i in range(num_im):
        if datadict['class_labels'][i] not in seen and datadict['class_labels'][i] not in unseen:
            continue
        for m in range(num_attr):
            X_attr[i, m] = (attr_weights[m, :] * X[i, :].T)[0, 0]
    
    mean_dict, means = {}, []
    covariance_dict, covariances = {}, []
    
    for seen_cat in seen:
        data = X_attr[datadict['class_labels'] == seen_cat, :]
        if data.shape[0] == 0:
            continue
        gmm = GMM().fit(data)
        mean_dict[seen_cat] = gmm.means_[0]
        covariance_dict[seen_cat] = gmm.covariances_[0]

        means.append(gmm.means_[0].tolist())
        covariances.append(gmm.covariances_[0].tolist())
    means = np.array(means)
    means = np.sort(means, axis = 0)

    dms = np.zeros((num_attr,))
    for i in range(1, len(seen)):
        dms += means[i, :] - means[i - 1, :]
    dms = dms / (len(seen))
    covariances = np.array(covariances)
    mean_covars = np.mean(covariances, axis = 0)
    X_attr = X_attr.reshape((X_attr.shape[0],))
    ## normalize X_attr between 0 and 1
    X_attr = np.round((X_attr - X_attr.min()) / (X_attr.max() - X_attr.min())* 100, 3)
    ## save datadict features paths and corresponding X_attr as csv
    id2label = {v: k for k, v in emotion_dict.items()}
    target_mean = None
    # message = ''
    message = {id2label[seen[0]]: {}, id2label[seen[1]]: {}, 'outliers': '# of neutral samples predicted as High intensity: '}
    for emt_idx in seen:
        score = X_attr[datadict['class_labels'] == emt_idx]
        message[id2label[emt_idx]]['mean'] = score.mean()
        message[id2label[emt_idx]]['var'] = score.var()
        message[id2label[emt_idx]]['min'] = score.min()
        message[id2label[emt_idx]]['max'] = score.max()

        # message += f'{id2label[emt_idx]} Mean: {score.mean()}\n'
        # message += f'{id2label[emt_idx]} Var: {score.var()}\n'
        # message += f'{id2label[emt_idx]} min/max: {score.min()} {score.max()}\n'
        if emt_idx == emotion_dict[args.emotion]:
            target_mean = score.mean()
    df = pd.DataFrame({'feature_paths': datadict['feature_paths'], 'X_attr': X_attr})
    df['bin_label'] = df['X_attr'].apply(lambda x: 1 if x > target_mean else 0)
    outlier_count = df.apply(lambda x: 1 if x['bin_label'] == 1 and 'Neutral' in x['feature_paths'] else 0, axis=1)
    message['outliers'] += str(outlier_count.sum())
    # message += f'Number of images in bin_label == 1 and emotion == Neutral: {outlier_count.sum()}\n\n'
    df['bin_label'] = df.apply(lambda x: 0 if 'Neutral' in x['feature_paths'] else x['bin_label'], axis=1)
    df.to_csv(os.path.join(results_dit, args.speaker, f'{args.emotion}.csv'), index=False)
    # with open(os.path.join(results_dit, args.speaker, f'{args.speaker}.txt'), 'a') as f:
    #     f.write(message)
    tgt_emotion = args.emotion if args.emotion in emotion_dict.keys() else 'All'
    message_path = os.path.join(results_dit, args.speaker, f'{args.speaker}_{tgt_emotion}.json')
    with open(message_path, 'w') as f:
        json.dump(message, f, indent=4)

# python learned_rank_func/training.py --dataset IMOCap
# python learned_rank_func/training.py --dataset ESD
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker', type=str, default=None)
    parser.add_argument('--emotion', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ESD', help='ESD or IMOCap')
    args = parser.parse_args()
    if args.dataset == 'ESD':
        from configs.esd_config import *
    elif args.dataset == 'IMOCap':
        from configs.imocap_config import *
    collated_datadict = np.load(datadict_path, allow_pickle=True).item()
    
    speaker_set = np.unique(collated_datadict['speakers'])
    if args.speaker not in speaker_set:
        for speaker in speaker_set:
            args.speaker = speaker
            os.makedirs(os.path.join(weights_dit, args.speaker), exist_ok=True)
            os.makedirs(os.path.join(results_dit, args.speaker), exist_ok=True)
            
            for emotion in emotion_dict.keys():
                if emotion == 'Neutral':
                    continue
                args.emotion = emotion
                score_path = os.path.join(results_dit, args.speaker, f'{args.emotion}.csv')
                weight_path = os.path.join(weights_dit, args.speaker, f'weights_{args.emotion}.npy')
                if os.path.exists(score_path):
                    print(f'{args.speaker}/{args.emotion}.csv already exists. Skipping predicting.')
                    continue
                datadict = load_data(args)
                
                if not os.path.exists(weight_path):
                    print(f'{args.speaker}/{args.emotion}.npy does not exist. Training.')
                    X, S, O, C_S, C_O = create_O_S_mat(datadict)
                    w = rank_svm(X, S, O, C_S, C_O)
                    np.save(weight_path, w)
                else:
                    print(f'{args.speaker}/{args.emotion}.npy already exists. Skipping training.')
                    w = np.load(weight_path)
                eval(args, datadict)
    else: 
        datadict = load_data(args, collated_datadict)
        X, S, O, C_S, C_O = create_O_S_mat(datadict)
        
        if os.path.exists(os.path.join(weights_dit, f'weights_{args.speaker}_{args.emotion}.npy')):
            print(f'weights_{args.speaker}_{args.emotion}.npy already exists. Skipping training.')
            eval(args, datadict)
        else:
            w = rank_svm(X, S, O, C_S, C_O)
            np.save(os.path.join(weights_dit, f'weights_{args.speaker}_{args.emotion}'), w)
            eval(args, datadict)


