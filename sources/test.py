import matplotlib.pyplot as plt
import os
import json
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tgt
# import tiktoken
import pandas as pd
import numpy as np
# import plotly.graph_objects as go
# from utils.eval_utils import read_lexicon
import string
from text import text_to_sequence
from string import punctuation
import re
from g2p_en import G2p
from tqdm import tqdm

from utils.pitch_tools import denorm_f0

# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.encoding_for_model(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

def gen_XY(src_path):
    with open(src_path, 'r') as f:
        lines = f.readlines()
        ## filter lines start with 'Validation Step'
        lines = [line for line in lines if line.startswith('Validation Step')]

    steps, losses, accs = [], [], []
    for line in lines:
        line = line.strip().split(',')
        steps.append(int(line[0].split(':')[1]))
        losses.append(float(line[1].split(':')[1]))
        accs.append(float(line[2].split(':')[1]))
    return steps, losses, accs

def plot_loss(src_path):
    # pooling_modes = ['AttnStats', 'Attn', 'max']
    # for pooling_mode in pooling_modes:
    #     src_path = f'./Audio_Data/output/ESD_{pooling_mode}/log/val/log.txt'
    #     plot_loss(src_path)
    with open(src_path, 'r') as f:
        lines = f.readlines()
        ## filter lines start with 'Validation Step'
        lines = [line for line in lines if line.startswith('Validation Step')]
    steps, losses, accs = [], [], []
    for line in lines:
        line = line.strip().split(',')
        steps.append(int(line[0].split(':')[1]))
        losses.append(float(line[1].split(':')[1]))
        accs.append(float(line[2].split(':')[1]))
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(steps, losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.plot(steps, accs)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig(src_path.replace('log.txt', 'loss.png'), dpi = 720)

def move_prompt(src_dit, tgt_dit):
    # move_prompt(src_dit='./Audio_Data/output/ESD_unsup_ph_wi/result/450000/st_eval_/', 
    # tgt_dit='./Audio_Data/output/ESD_unsup_ph_wi/result/450000/st_eval/')
    for raw_text in os.listdir(src_dit):
        if not os.path.isdir(os.path.join(src_dit, raw_text)):
            continue
        src_path = os.path.join(src_dit, raw_text)
        tgt_path = os.path.join(tgt_dit, raw_text)
        
        os.makedirs(tgt_path, exist_ok=True)
        for prompt in os.listdir(src_path):
            if prompt.endswith('.txt'):
                src_prompt = os.path.join(src_path, prompt)
                prompt_seg = prompt.split('_')
                del prompt_seg[-2]
                prompt = '_'.join(prompt_seg)
                # breakpoint()
                tgt_prompt = os.path.join(tgt_path, prompt)
                os.system(f'cp {src_prompt} {tgt_prompt}')

def clean_dir(src_dit):
    # clean_dir(src_dit='./Audio_Data/output/ESD_unsup_ph_wi/result/450000/obj_eval/')
    for raw_text in os.listdir(src_dit):
        if not os.path.isdir(os.path.join(src_dit, raw_text)):
            continue
        src_path = os.path.join(src_dit, raw_text)
        for mis in os.listdir(src_path):
            if not mis.endswith('.txt'):
                tgt_path = os.path.join(src_path, mis)
                # os.remove(tgt_path)
                if os.path.isdir(tgt_path):
                    os.system(f'rm -rf {tgt_path}')
                else:
                    os.system(f'rm {tgt_path}')

def populate(valset, op='val_mos.txt'):
    speakers = ['0011','0012','0013','0014','0015','0016','0017','0018','0019','0020']
    emotion_pointer = {'Angry': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0}
    with open(valset, 'r') as f:
        lines = f.readlines()
    r_lines = []
    for line in lines:
        line = line.strip().split('|')
        emotion = line[0]
        line[0] = line[0]+'_'+str(emotion_pointer[emotion])
        emotion_pointer[emotion] += 1
        r_lines.append(line)
    f = open(op, 'w')
    for line in r_lines:
        for speaker in speakers:
            c_line = line.copy()
            c_line[0] = speaker+'_'+c_line[0]
            c_line.insert(1, speaker)
            f.write('|'.join(c_line)+'\n')
    f.close()

def st_ref(src_dit):
    # src_dit = '/home/shaozuo/etts_prj/Daft-Exprt/Audio_Data/preprocessed_data/ESD_unsup/relative_attr/labels'
    # st_ref(src_dit)
    ref_dict = {}
    for spker in sorted(os.listdir(src_dit)):
        spker_path = os.path.join(src_dit, spker)
        ref_dict[spker] = {}
        for emt in sorted(os.listdir(spker_path)):
            if emt.endswith('.csv'):
                emt_path = os.path.join(spker_path, emt)
                emt_cat = emt.split('.')[0]
                ref_dict[spker][emt_cat] = {}
                df = pd.read_csv(emt_path)
                df, neu_df = df[df['feature_paths'].str.contains(emt_cat)], df[df['feature_paths'].str.contains('Neutral')]
                df.reset_index(drop=True, inplace=True)
                max_idx, min_idx = df['X_attr'].idxmax(), df['X_attr'].idxmin()
                mean = df['X_attr'].mean()
                mean_idx = (df['X_attr']-mean).abs().idxmin()
                neu_idx = neu_df['X_attr'].idxmin()
                max_id = os.path.basename(df.iloc[max_idx]['feature_paths']).split('.')[0] + f'_{emt_cat}'
                min_id = os.path.basename(df.iloc[min_idx]['feature_paths']).split('.')[0] + f'_{emt_cat}'
                mean_id = os.path.basename(df.iloc[mean_idx]['feature_paths']).split('.')[0] + f'_{emt_cat}'
                neu_id = os.path.basename(neu_df.iloc[neu_idx]['feature_paths']).split('.')[0] + f'_Neutral'
                ref_dict[spker][emt_cat]['High'] = max_id
                ref_dict[spker][emt_cat]['Low'] = min_id
                ref_dict[spker][emt_cat]['Medium'] = mean_id
                ref_dict[spker][emt_cat]['Neutral'] = neu_id
    with open('ref.json', 'w') as f:
        json.dump(ref_dict, f)

def plot_mos_response(response_path):
    df = pd.read_csv(response_path)
    scores = {'summary': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
              'Angry': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
              'Happy': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
              'Sad': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
              'Surprise': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []}}
    map_tool = {'1': 'bsl', '2': 'emt', '3': 'pmt', '4': 'pmt_l'}
    num_qns = int(len(df.columns[2:-4])/4)
    num_participants = len(df)
    for wav_id in df.columns[2:-4]:
        text_id, frw = int(wav_id.split('_')[0]), wav_id.split('_')[1]
        if text_id in range(0,5):
            emotion = 'Angry'
        elif text_id in range(5,10):
            emotion = 'Happy'
        elif text_id in range(10,15):
            emotion = 'Sad'
        else:
            emotion = 'Surprise'
        scores[emotion][map_tool[frw]].extend(df[wav_id].tolist())
        scores['summary'][map_tool[frw]].extend(df[wav_id].tolist())
    stats_dict = {'Angry': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
             'Happy': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []}, 
             'Sad': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []},
             'Surprise': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []}}
    for emotion in scores.keys():
        if emotion == 'summary':
            continue
        else:
            for frw in scores[emotion].keys():
                ## remove least 5 values in scores[emotion][frw]
                scores[emotion][frw].sort()
                scores[emotion][frw] = scores[emotion][frw][25:]
                ## normalize the score to -2 to 2
                # import pdb; pdb.set_trace()
                # scores[emotion][frw] = [round((x - min(scores[emotion][frw]))/(max(scores[emotion][frw]) - min(scores[emotion][frw]))*4-2, 2) for x in scores[emotion][frw]]
                mean = np.mean(scores[emotion][frw])
                stats_dict[emotion][frw].append(mean)
                ci = stats.t.interval(0.95, len(scores[emotion][frw])-1, loc=mean, scale=stats.sem(scores[emotion][frw]))
                heights = round((ci[1] - ci[0])/2,2)
                stats_dict[emotion][frw].append(heights)
    cats = ['Angry', 'Happy', 'Sad', 'Surprise', 'Angry']
    i = 0
    bars1 = [stats_dict['Angry']['bsl'][i], stats_dict['Happy']['bsl'][i], stats_dict['Sad']['bsl'][i], stats_dict['Surprise']['bsl'][i], stats_dict['Angry']['bsl'][i]]
    bars2 = [stats_dict['Angry']['emt'][i], stats_dict['Happy']['emt'][i], stats_dict['Sad']['emt'][i], stats_dict['Surprise']['emt'][i], stats_dict['Angry']['emt'][i]]
    bars3 = [stats_dict['Angry']['pmt_l'][i], stats_dict['Happy']['pmt_l'][i], stats_dict['Sad']['pmt_l'][i], stats_dict['Surprise']['pmt_l'][i], stats_dict['Angry']['pmt_l'][i]]
    bars4 = [stats_dict['Angry']['pmt'][i], stats_dict['Happy']['pmt'][i], stats_dict['Sad']['pmt'][i], stats_dict['Surprise']['pmt'][i], stats_dict['Angry']['pmt'][i]]
    fig = go.Figure()
    fill_value = 'tonext'
    # breakpoint()
    fig.add_trace(
        go.Scatterpolar(r=bars1, theta=cats, name='Daft-Exprt(Baseline)', fill=fill_value, 
                        line = dict(color='rgba(135, 206, 235, 0.5)', width=3)))
    fig.add_trace(
        go.Scatterpolar(r=bars2, theta=cats, name='FS2-w/o-Prompt', fill=fill_value, 
                        line=dict(color='rgba(0, 191, 255, 0.5)', width=3)))
    fig.add_trace(
        go.Scatterpolar(r=bars3, theta=cats, name='FS2-L-Prompt', fill=fill_value, 
                        line=dict(color='rgba(30, 144, 255, 0.5)', width=3))) #
    fig.add_trace(
        go.Scatterpolar(r=bars4, theta=cats, name='FS2-G&L-Prompt', fill=fill_value, 
                        line=dict(color='rgba(0, 0, 255, 0.5)', width=3)))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[2, 4],
                tickvals=[2, 2.5, 3, 3.5, 4],
                tickfont = dict(size = 10, family = 'Times New Roman'),
                showline=False
            )),
        showlegend=True
    )
    fig.update_polars(
        angularaxis = dict(
            tickfont = dict(size = 16, family = 'Times New Roman')
        )
    )
    fig.show()
    '''The following code is ploting bar chart'''
    # barWidth = 0.15
    # # bars1 = [,9,2]
    # i = 0
    # bars1 = [stats_dict['Angry']['bsl'][i], stats_dict['Happy']['bsl'][i], stats_dict['Sad']['bsl'][i], stats_dict['Surprise']['bsl'][i]]
    # bars2 = [stats_dict['Angry']['emt'][i], stats_dict['Happy']['emt'][i], stats_dict['Sad']['emt'][i], stats_dict['Surprise']['emt'][i]]
    # bars3 = [stats_dict['Angry']['pmt_l'][i], stats_dict['Happy']['pmt_l'][i], stats_dict['Sad']['pmt_l'][i], stats_dict['Surprise']['pmt_l'][i]]
    # bars4 = [stats_dict['Angry']['pmt'][i], stats_dict['Happy']['pmt'][i], stats_dict['Sad']['pmt'][i], stats_dict['Surprise']['pmt'][i]]
    # i = 1
    # yer1 = [stats_dict['Angry']['bsl'][i], stats_dict['Happy']['bsl'][i], stats_dict['Sad']['bsl'][i], stats_dict['Surprise']['bsl'][i]]
    # yer2 = [stats_dict['Angry']['emt'][i], stats_dict['Happy']['emt'][i], stats_dict['Sad']['emt'][i], stats_dict['Surprise']['emt'][i]]
    # yer3 = [stats_dict['Angry']['pmt_l'][i], stats_dict['Happy']['pmt_l'][i], stats_dict['Sad']['pmt_l'][i], stats_dict['Surprise']['pmt_l'][i]]
    # yer4 = [stats_dict['Angry']['pmt'][i], stats_dict['Happy']['pmt'][i], stats_dict['Sad']['pmt'][i], stats_dict['Surprise']['pmt'][i]]
    # # The x position of bars
    # cmap = plt.cm.Blues
    # colors = [cmap(i) for i in np.linspace(0.3, 1, 4)]

    # r1 = np.arange(len(bars1))
    # r2 = [x + barWidth for x in r1]
    # r3 = [x + barWidth for x in r2]
    # r4 = [x + barWidth for x in r3]
    # plt.figure(figsize=(12,6))
    # plt.bar(r1, bars1, width = barWidth, color = colors[0], edgecolor = 'black', yerr=yer1, capsize=5, label='Daft-Exprt(Baseline)')
    # plt.bar(r2, bars2, width = barWidth, color = colors[1], edgecolor = 'black', yerr=yer2, capsize=5, label='FS2-w/o-Prompt')
    # plt.bar(r3, bars3, width = barWidth, color = colors[2], edgecolor = 'black', yerr=yer3, capsize=5, label='FS2-L-Prompt')
    # plt.bar(r4, bars4, width = barWidth, color = colors[3], edgecolor = 'black', yerr=yer4, capsize=5, label='FS2-G&L-Prompt')
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)

    # # general layout
    # plt.xticks([r + barWidth for r in range(len(bars1))], ['Neu-Ang', 'Neu-Hap', 'Neu-Sad', 'Neu-Sur'])
    # plt.ylabel('height')
    # plt.show()
    # plt.savefig('mos_scores.png', dpi = 1080)

def process_mos_response(response_path):
    df = pd.read_csv(response_path)
    scores = {'summary': {'bsl': [], 'emt': [], 'pmt': [], 'pmt_l': []}}
    map_tool = {'1': 'bsl', '2': 'emt', '3': 'pmt', '4': 'pmt_l'}
    num_qns = int(len(df.columns[2:-4])/4)
    num_participants = len(df) 
    for wav_id in df.columns[2:-4]:
        text_id, frw = wav_id.split('_')[0], wav_id.split('_')[1]
        if text_id not in scores:
            scores[text_id] = {'bsl': 0, 'emt': 0, 'pmt': 0, 'pmt_l': 0}
        scores[text_id][map_tool[frw]] += df[wav_id].sum()
        # scores['summary'][map_tool[frw]] += df[wav_id].sum()    
        scores['summary'][map_tool[frw]].append(df[wav_id].sum())
    barWidth = 0.3
    bars1 = [10,9,2]
    bars2 = [10.8, 9.5, 4.5]
    # Choose the height of the error bars (bars1)
    yer1 = [0.5, 0.4, 0.5]
    
    # Choose the height of the error bars (bars2)
    yer2 = [1, 0.7, 1]
    
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    # Create blue bars
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='poacee')
    
    # Create cyan bars
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='sorgho')
    
    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], ['cond_A', 'cond_B', 'cond_C'])
    plt.ylabel('height')
    plt.legend()
    plt.show()
    plt.savefig('mos_scores.png', dpi = 1080)
    # for k in scores.keys():
    #     if k == 'summary':
    #         for kk in scores[k].keys():
    #             scores[k][kk] /= num_qns*num_participants
    #     else:
    #         for kk in scores[k].keys():
    #             scores[k][kk] /= num_participants
    
    # with open('mos_scores.json', 'w') as f:
    #     json.dump(scores, f)

def preprocess_english(word_df):
    lexicon = read_lexicon("lexicon/librispeech-lexicon.txt")
    g2p = G2p()
    phones = ''
    words = word_df['word'].tolist()
    words = [''.join([c for c in w if c not in punctuation]) for w in words]
    phone_lens = []
    for w in words:
        if w.lower() in lexicon:
            word_phones = lexicon[w.lower()]
        else:
            word_phones = list(filter(lambda p: p != " ", g2p(w)))
        word_phones = "{" + " ".join(word_phones) + "}"
        word_phones = re.sub(r"\{[^\w\s]?\}", "{sp}", word_phones)
        phone_lens.append(len(word_phones.split(' ')))
        phones += word_phones
    phones = phones.replace("}{", " ")
    phones = phones[1:-1].split(' ')
    phones = ['{'+p+'}' for p in phones]
    # print("Raw Text Sequence: {}".format(words))
    # print("Phoneme Sequence: {}".format(phones))
    # sequence = np.array(
    #     text_to_sequence(
    #         phones, ["english_cleaners"]
    #     )
    # )
    return phones

def read_factors(prompt_output):
    utterance_df, word_df = None, None
    current_df = None
    for idx, line in enumerate(prompt_output):
        line = line.strip().lower()
        if line.startswith("|"):
            values = [value.strip() for value in line.split("|") if value.strip()]
            # print(values)
            if len(values) == 3:
                current_df = "utterance"
            elif len(values) == 4:
                current_df = "word"
            if "---" in "".join(values):
                continue
            if current_df == "utterance" and utterance_df is None:
                utterance_df = pd.DataFrame(columns=values)
            elif current_df == "word" and word_df is None:
                word_df = pd.DataFrame(columns=values)
            else:
                # Append data to the respective dataframe
                if current_df == "utterance":
                    utterance_df.loc[len(utterance_df)] = values
                elif current_df == "word":
                    if len(values) != 4:
                        print('Num of columns mismatch in line: ', idx, line)
                        continue
                    elif values[0] in string.punctuation:
                        continue
                    ### control durations within (-2,2)
                    values[-1] = str(max(-2, min(2, int(values[-1]))))
                    word_df.loc[len(word_df)] = values
    phone_sequence = preprocess_english(word_df)

    return phone_sequence     
   
def plot_pe(pe_path):
    f0_denorm = {}
    for text_id in tqdm(sorted(os.listdir(pe_path))):
        if text_id not in ['Angry_1', 'Happy_6', 'Sad_11', 'Surprise_14']:
            continue

        sub_id = os.path.join(pe_path, text_id)
        # emotion = text_id.split('_')[0]
        f0_denorm[text_id] = {}
        with open(os.path.join(sub_id, f'{text_id}_prompt.txt'), 'r') as f:
            prompt_output = f.readlines()
        phones = read_factors(prompt_output)
        f0_denorm[text_id]['phones'] = phones
        for stat in os.listdir(sub_id):
            if not stat.endswith('.json'):
                continue
            stat_path = os.path.join(sub_id, stat)
            st_lvl = stat.split('.')[0].split('_')[-1]
            with open(stat_path, 'r') as f:
                stat_dict = json.load(f)
            # pitch[st_lvl] = stat_dict['pitch']
            f0_denorm[text_id][st_lvl] = stat_dict['f0_denorm']

            # energy[st_lvl] = stat_dict['energy']
    with open('f0_denorm.json', 'w') as f:
        json.dump(f0_denorm, f)
    count = 0
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(44, 14))
    count = 0
    f2b = {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}
    
    light_red, light_green, light_blue, light_org = '#FFCCCC', '#90EE90', '#ADD8E6', '#D8BFD8'
    background_colors = [
                        (1, 0.8, 0.8, 0.5),  # light red
                        (0.8, 1, 0.8, 0.5),  # light green
                        (0.8, 0.8, 1, 0.5),  # light blue
                        (1, 0.8, 1, 0.5)     # light purple
                        ]
    # background_colors = [light_red, light_green, light_blue, light_org] 
    for k,v in f0_denorm.items():
        # plt.figure(figsize=(20,10))
        emt = k.split('_')[0]
        phones = v['phones']
        r, c = f2b[count][0], f2b[count][1]
        axs[r][c].plot(v['H'], label = 'High', linewidth = 4)
        axs[r][c].plot(v['M'], label = 'Medium', linewidth = 4)
        axs[r][c].plot(v['L'], label = 'Low', linewidth = 4)
        # if r == 1:
        #     axs[r][c].set_xlabel('Phonemes', fontweight="bold", fontsize = 48)
        # if c == 0:
        #     axs[r][c].set_ylabel('F0 (Hz)', fontweight="bold", fontsize = 48)
        # axs[r][c].set_xlabel('Phonemes', fontweight="bold")
        axs[r][c].set_xticks(range(len(phones)), phones, fontsize = 16)
        # axs[r][c].set_ylabel(k)
        # axs[r][c].legend(loc = 'best')
        axs[r][c].grid()
        # axs[r][c].set_title(f'{emt}', fontsize = 36, fontweight="bold", fontname = 'Times New Roman')
        axs[r][c].set_facecolor(background_colors[count])
        count += 1
    handles, labels = axs[r][c].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize = 36)
    # plt.suptitle('Pitch Contour', fontweight="bold")
    plt.savefig(f'pitch_contour.png', dpi = 720)
    plt.close()
    # breakpoint()

def process_bws_response(response_path):
    df = pd.read_csv(response_path)
    scores = {'GT': {'summary': [], 'Angry': [], 'Happy': [], 'Sad': [], 'Surprise': []},
              'test': {'summary': [], 'Angry': [], 'Happy': [], 'Sad': [], 'Surprise': []}}
    answer_map = {'1': 1, '2': 3, '3': 2}
    for qn_id in sorted(df.columns[2:-4]):
        # text_id, frw = wav_id.split('_')[0], wav_id.split('_')[1]
        emotion = qn_id.split('_')[0][8:]
        int_lvl = qn_id.split('_')[-1]
        ans = df[qn_id].tolist()
        scores['test'][emotion].extend(ans)
        scores['test']['summary'].extend(ans)
        scores['GT'][emotion].extend([answer_map[int_lvl]]*len(ans))
        scores['GT']['summary'].extend([answer_map[int_lvl]]*len(ans))
        
        scores['test'][emotion].extend([answer_map[int_lvl]]*3)
        scores['test']['summary'].extend([answer_map[int_lvl]]*3)
        scores['GT'][emotion].extend([answer_map[int_lvl]]*3)
        scores['GT']['summary'].extend([answer_map[int_lvl]]*3)
    
    # f = open('bws_cm.json', 'w')
    # save_js = {}
    # for k in scores['GT'].keys():
    #     cr = classification_report(scores['GT'][k], scores['test'][k], output_dict=True)
    #     save_js[k] = cr
    # f.write(json.dumps(save_js, indent=4))
    # f.close()
    cms = {}
    for k in scores['GT'].keys():
        cr = confusion_matrix(scores['GT'][k], scores['test'][k])
        cms[k] = cr
    disp = ConfusionMatrixDisplay(cms, display_labels=['High', 'Mid', 'Low'])
    disp.plots(values_format='d')
    plt.savefig('cms.png', dpi = 1080)
    plt.close()

if __name__ == "__main__":
    # valset = 'val_bws_raw.txt'
    # populate(valset, op='val_bws.txt')

    # response_path = 'mos_data.csv'
    # plot_mos_response(response_path)
    # src_dit = '/home/shaozuo/etts_prj/Daft-Exprt/Audio_Data/preprocessed_data/ESD_unsup/relative_attr/labels'
    # st_ref(src_dit)

    # pe_path = '/home/shaozuo/etts_prj/Emo-CT-TTS/Audio_Data/output/ESD_unsup_ph_wi/result/450000/bws_eval'
    # plot_pe(pe_path)

    # response_path = 'sub_eval/bws_data.csv'
    # process_bws_response(response_path)

    # move_prompt(src_dit='./Audio_Data/output/ESD_unsup_ph_wi/result/450000/obj_eval/', 
    #             tgt_dit='./Audio_Data/output/ESD_LJ/result/900000/obj_eval/')

    move_prompt(src_dit='./Audio_Data/output_close/ESD_unsup_ph_wi/result/450000/obj_eval/', 
                tgt_dit='Audio_Data_partial/output/ESD_Emo_Int_unsup_ph/result/450000/obj_eval')
    


