import datetime
from logging import log
from pickle import NONE
from re import S, T
from matplotlib.pyplot import axis
import torch
import pandas as pd
import os
import numpy as np
import librosa
from utils.consts import PICK_POSE, SPEAKERS_CONFIG
from transformers import BertTokenizer



class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, type_data) -> None:
        assert(type_data in ['train', 'dev', 'test'])
        super().__init__()
        self.args = args
        self.speaker = args.speaker
        self.base_path = args.base_path
        self.type_data = type_data
        df = pd.read_csv(os.path.join(args.base_path, 'train.csv'))
        df = df[df['speaker'] == args.speaker]
        self.speaker_config = SPEAKERS_CONFIG[args.speaker]
        self.df = df[df['dataset'] == type_data]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.data = []
        # for i in range(len(df)):
        #     pose_fn = self.df.iloc[i]['pose_fn']
        #     data = np.load(os.path.join(self.base_path, pose_fn))
        #     self.data.append(data)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        pose_fn = self.df.iloc[index]['pose_fn']
        data = np.load(os.path.join(self.base_path, pose_fn))
        # data = self.data[index]
        # print(data['imgs'])
        # if self.audio_flag == 'raw':
        #     audio = data['audio']
        #     # print(audio.shape[0])
        #     if len(audio.shape) == 0 or audio.shape[0] == 0:
        #         logmel = data['audio_log_mel_400']
        #         # hop = 442 // 256
        #         pick = range(0, 256)
        #         logmel = logmel[pick, ]
        #     else:
        #         mel = librosa.feature.melspectrogram(
        #             audio,
        #             sr=16000,
        #             hop_length=260,
        #             n_fft=400,
        #             fmin=125,
        #             fmax=7500,
        #             n_mels=64,
        #             center=False,
        #         )
        #         # mel = mel.astype(np.float64)
        #         logmel = librosa.power_to_db(mel)
        #         mel_len = logmel.shape[1]
        #         # print(logmel.shape)
        #         logmel = logmel[:, : mel_len // 4 * 4]
        #         logmel = logmel.reshape(64 * 4, mel_len // 4)
        #         if logmel.shape[1] != 64:
        #             logmel = np.c_[logmel, np.zeros((256, 64 - logmel.shape[1]))]
                # print(logmel.shape)
            # logmel = logmel.transpose()
            
        logmel = data['audio_log_mel_400']

        # else:
        #     logmel = data['audio_log_mel_512']
            # print(logmel.shape)
        # if self.text_flag == 'w2v':
        #     text = data['text_w2v']
        # else:

        ### Change to token(need changing back)---------
        if self.args.model == 'multimodal_context':
            text = []
            tokens = data['text_token']
            for token in tokens:   
                word = self.tokenizer.decode([token])
                text.append(word)
            
        else: 
            text = data['text_bert']
        # text = np.expand_dims(text, axis=1)
        ###-------------------


        # print(logmel.shape)
        pose = data['pose']
        ###frames * 72
        pose = pose[:, PICK_POSE]
        mean = self.speaker_config['mean'][PICK_POSE]
        std = self.speaker_config['std'][PICK_POSE]
        pose = (pose - mean) / (std + np.finfo(float).eps)
        #############################################################
        ### maybe can be add: 1.norm(need to analyse data)      #######(two mean and var, )
        ###                   2.smooth(survey method)           #####    jihe  wuli?
        ############################################################# need to analyse the data distribution

        return {'audio': logmel, 'text': text, 'pose': pose}
