# Prompt-Driven Text-to-Speech Synthesis Based on Emotion & Intensity Control - PyTorch Implementation

## Instruction Manual

### Environment

It is suggested to use mini conda environment with PyTorch installed. You can install PyTorch via the official [link]() based on your system configuration.

You can install the Python dependencies by the commands below:

```yaml
# Config conda environment
conda update conda
conda create -n proemo python=3.8
conda activate proemo
# Install PyTorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
# Install the required libraries
pip install -r requirements.txt
```

### Data Download

Download LibriTTS, LJSpeech, ESD, IMOCap dataset from the following links:
- [LibriTTS](https://www.openslr.org/60/)
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
- [ESD]()
- [IMOCap]()
Download openSMILE toolkit

### Learned Ranked Function
We use learnd ranked function to retrieve the intensity scores from emotional dataset. To train the ranked function, you can run the following command:(DATASET could be ESD, IMOCap)
```yaml
python learned_rank_func/preprocess_os.py --dataset DATASET
python learned_rank_func/training.py --dataset DATASET
```
This could take around 1 hour. Alternatively, you may just use the ready-made intensity scores in Preprocessed_Data/DATASET/relative_attr. Make sure this folder exists with valid files to proceed data preprocessing
### Dataset Preprocess
This framework supports LibriTTS, LJSpeech, ESD, IMOCap. The duration is modeld unsupervisedly, the pitch/energy/duration information are in phoneme-level. You can just run the following command: 
```yaml
python prepare_align.py --dataset DATASET (cleared)
```
Usually this process is for Montreal Forced Aligner(MFA), but here the purpose is just to format the dataset in a consistent way. Then run the following command to preprocess the dataset:
```yaml
python3 preprocess.py --dataset DATASET (cleared)
```

### Training
Train the model with three choices (cleared)
```yaml
### Pretrain on LibriTTS
python train.py --dataset LibriTTS 
### Fine-tune the pretrained model on ESD
python train.py --dataset ESD --restore_step 900000 --checkpoint LibriTTS --emotion_label 1 --intensity_label 1
### or fine-tune the pretrained model on IMOCap
python train.py --dataset IMOCap --restore_step 900000 --checkpoint LibriTTS --emotion_label 1 --intensity_label 1
```
For emotional datasets, emotion_label should be 1, the intensity_label can be 0 or 1

### Monitor the training
You can monitor the training process by running the following command:
```yaml
tensorboard --logdir Audio_Data/output/CHECKPOINT --port 4000 --host 0.0.0.0
tensorboard --logdir Audio_Data/output/CHECKPOINT --port 4000 --bind_all
```
It will create a link to access the tensorboard. CHECKPOINT is the name of the folder in Audio_Data/output, port number could vary based on your needs. host 0.0.0.0 will create a localhost, bind_all will create a sharable link.

### Sythesize audios
Before you synthesize audios, please make sure the CHECKPOINT/results directory contains the emotion representations and intensity representations, for all the data samples. If not, please run the following command:
```yaml
python3 sythesize_embedding.py -- TODO
```
Alternatively, you can always take the read-made embeddings from the [link]()
You can execute the objective evaluation and subjective evalution to obtain sythesized audios accross different settings. The objective evaluation is reconstructed from the validation set, while the subjective evaluation is reconstructed from any text. You can run the following command:
```yaml
python objective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --checkpoint Audio_Data/output/CHECKPOINT --dataset DATASET --emotion_label 1 --intensity_label 0
python objective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --checkpoint Audio_Data/output/CHECKPOINT --dataset DATASET --emotion_label 1 --intensity_label 1
```
The scripts are designed for ESD and IMOCap dataset, but IMOCap checkpoints are not available yet due to performance issue. 
You may also run the subjective evaluation by running the following command:
```yaml
python subjective_eval.py --source val_mos.txt --restore_step 450000 --mode batch --checkpoint ESD --label emotion
```
val_mos.txt, val_bws.txt is free-form any input text, while val_prompt_unsup.txt is from the validation set. The checkpoint is the name of the folder in Audio_Data/output.

```yaml
# python subjective_eval.py --source val_bws.txt --restore_step 450000 --mode batch --checkpoint ESD --label intensity
python subjective_eval.py --source Audio_Data/preprocessed_data/ESD_unsup/val_prompt_unsup.txt --restore_step 450000 --mode batch --checkpoint ESD --label intensity
```

## TODO
add synthesize embedding
fix subjective evaluation
fix any text generation of bws(optional)
fix wer whisper
add the Audio_Data folder and include necessary files
add credits and collaborators
