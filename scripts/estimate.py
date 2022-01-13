from logging import log
import sys
[sys.path.append(i) for i in ['.', '..']] ### Add search path of 'import'


import torch
import os
import resampy
from scipy.io import wavfile
import subprocess
from data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from model.baseline import Base, Twostage
import argparse
import librosa
import h5py
from model.multimodal_context_net import PoseGenerator
import pickle
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pick_pose = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]


parser = argparse.ArgumentParser(description='Description for estimation')
parser.add_argument('-pats','--pats_base_path', default='/mnt/lustressd/yuzhengming/gesture_data/pats/data/processed')

parser.add_argument('-l','--load_from_path', required=True)
parser.add_argument('-b','--base_path', required=True)
parser.add_argument('-s','--speaker', default='conan')
parser.add_argument("--name", type=str, required=True)
parser.add_argument('--model', required=True, default=None)
parser.add_argument('--n_pre_poses', type=int, default=4)
parser.add_argument('--n_poses', default=64)
parser.add_argument('--input_context', default='both')
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--wordembed_dim', type=int, default=300)
parser.add_argument('--dropout_prob', default=0.3)
parser.add_argument("--freeze_wordembed", type=bool, default=False)
parser.add_argument("--n_layers", type=int, default=4)


args = parser.parse_args()


### need: model, data with img, wav, 
### want to have: gt img | gt smpl | pred smpl, pred smpl + gt img
### 


fs_512 = int(45.6*1000/512) #int(44.1*1000/512) #112 #round(22.5*1000/512)
fs_400 = int(16.52 *1000/160)
window_512 = int(4.3 * fs_512)
window_400 = int(4.3 * fs_400)
fs_ratio_512 = round(fs_512 / 15)
fs_ratio_400 = round(fs_400 / 15)
window_hop_512 = int(64*fs_ratio_512)
window_hop_400 = int(64*fs_ratio_400)


SR = 16000
AUDIO_FN_TEMPLATE = os.path.join(args.base_path, '%s/train_with_token/audio/%s-%s-%s.wav')
TRAINING_SAMPLE_FN_TEMPLATE = os.path.join(args.base_path, '%s/train_with_token/npz/%s-%s-%s.npz')

def get_smpl_pose_path(base_path, speaker, pose_fn):
    new_pose_fn = pose_fn[:-4] + '.h5'
    return os.path.join(base_path, speaker, 'pose_smpl', new_pose_fn)


def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)

def raw_repr(path, sr=None):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return wav, sr

def get_video_path(base_path, speaker, video_fn):
    return os.path.join(base_path, speaker, 'videos', video_fn)

def get_pats_path(base_path, speaker, interval_id):
    interval_fn = str(interval_id) + '.h5'
    return os.path.join(base_path, speaker, interval_fn)

def save_audio_sample_from_video(vid_path, audio_out_path, audio_start, audio_end, sr=44100):
    if not (os.path.exists(os.path.dirname(audio_out_path))):
        os.makedirs(os.path.dirname(audio_out_path))
    cmd = 'ffmpeg -i "%s" -ss %s -to %s -ab 160k -ac 2 -ar %s -vn "%s" -y -loglevel warning' % (
    vid_path, audio_start, audio_end, sr, audio_out_path)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__' :
    # print(dataset.__len__())#85964

    ###------------shorter sequense-------------------------------
    # df = pd.read_csv(os.path.join(args.base_path, 'train.csv'))
    # df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]

    # # with_meaning = [117549, 118502, 118983, 115789, 
    # # 115059, 119259, 114249, 117157, 118716, 115073, 115854,
    # # 115073, 119913, 113887, 114522, 117476, 119882, 114473, 115022, 114581,
    # # 116957, 117722, 114189, 119666] (train + dev)

    # # with_meaning = [114683, 115100, 115154, 116487, 118259, 118589, 118720, 
    # # 113999, 114385, 115743, 116078, 116465, 116686, 117776, 117828, 118578, 119531, 
    # # 113890, 114189, 114959, 116344, 118102, 119509, 113897,
    # # 116334, 116398, 117834, 117977, 118085, 114095, 116070, 116677,
    # # 116683, 119709, 114189, 117059, 117133, 113890, 114819, 115097, 116005, 
    # # 120132, 114021, 115823, 116410, 117269, 117310, 117712] (train + dev)

    # ##114189 in dev

    # ## in dev    
    # with_meaning = [116344, 114095, 113890, 118259, 116398, 114189, 116487, 118085, 114959, 116070, 119709]

    # checkpoint = torch.load(args.load_from_path)
    # generator = Base(args).to(device)
    # generator.load_state_dict(checkpoint['gen_dict'])


    # keypoints1_list = []
    # keypoints2_list = []
    # wav_list = []
    # imgs_list = []

    # # df_loss = []
    # for i, row in tqdm(df.iterrows(), total=len(df)):
    #     # try:
    #     interval_id = row['interval_id']

    #     if not (interval_id in with_meaning):
    #         continue

    #     # if len(keypoints1_list) >= 160:
    # #         break

    #     pose_fn = row['pose_fn']
    #     data = np.load(os.path.join(args.base_path, pose_fn))

    #     logmel = data['audio_log_mel_400']
    #     text = data['text_bert']
    #     # text = np.expand_dims(text, axis=1)

    #     logmel = np.expand_dims(logmel, 0)
    #     text = np.expand_dims(text, 0)

    #     logmel = torch.tensor(logmel)
    #     text = torch.tensor(text)
    #     logmel = logmel.float()
    #     text = text.float()

    #     logmel = logmel.to(device)
    #     text = text.to(device)

    #     wav = data['audio']
    #     imgs = data['imgs']
    #     if len(wav.shape) == 0 or wav.shape[0] == 0:
    #         continue

    #     keypoints2 = data['pose']
    #     keypoints2 = keypoints2[:, pick_pose]

    #     keypoints1 = generator(in_audio=logmel, in_text=text).cpu()
    #     keypoints1 = keypoints1.detach().numpy()
    #     keypoints1_list.append(keypoints1[0])
    #     keypoints2_list.append(keypoints2)
    #     wav_list.append(wav)
    #     imgs_list.append(imgs)


    #     # except Exception as e:
    #     #     # logger.exception(e)
    #     #     continue
    #     # df_loss.append(row_loss)

    # keypoints1_list = np.array(keypoints1_list) 
    # keypoints2_list = np.array(keypoints2_list)
    # print(len(wav_list))
    # wav_list = np.array(wav_list)
    # imgs_list = np.array(imgs_list)

    # name = 'base_norm'
    # save_path = '/mnt/lustressd/yuzhengming/visulization/conan/' + name +  '/slices_npz/dev.npz'
    # os.makedirs('/mnt/lustressd/yuzhengming/visulization/conan/' + name +  '/slices_npz')
    # np.savez(save_path, keypoints1_list=keypoints1_list, keypoints2_list=keypoints2_list, wav_list=wav_list, imgs_list=imgs_list)
    # print('saving over')
    # # np.array(df_loss)

    # main()


    ###-------------------generate longer sequence-----------###########
    count = 0

    df = pd.read_csv(os.path.join(args.base_path, "frames_df_10_19_19.csv"))
    if args.speaker is not None:
        df = df[df['speaker'] == args.speaker]
    df = df[df['dataset'] == 'dev']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # intervals_unique = df['interval_id'].unique()

    checkpoint = torch.load(args.load_from_path, map_location=torch.device(device))
    if args.model == 'baseline':
        generator = Base(args).to(device)
    elif args.model == 'multimodal_context':
        cache_path = os.path.join(os.path.split(args.base_path)[0], 'vocab_cache.pkl')
        print('    loaded from {}'.format(cache_path))
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)
        generator = PoseGenerator(args,
                            lang_model=lang_model,
                            n_words=lang_model.n_words,
                            word_embed_size=args.wordembed_dim,
                            word_embeddings=lang_model.word_embedding_weights,
                            z_obj=1,
                            pose_dim=45).to(device)
    elif args.model == 'Twostage':
        generator = Twostage(args).to(device)
    else:
        raise('error model type!')
    generator.load_state_dict(checkpoint['gen_dict'])


    

    intervals = df['interval_id'].unique()

    for interval in tqdm(intervals):
        keypoints1_list = []
        keypoints2_list = []
        wav_list = []
        imgs_list = []

        df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
        
        if len(df_interval) < 4 * 64:
            continue

        video_fn = df_interval.iloc[0]['video_fn']
        speaker_name = df_interval.iloc[0]['speaker']
        poses = np.array([h5py.File(get_smpl_pose_path(args.base_path, row['speaker'], row['pose_fn']), 'r')['poses'][:] for _, row in df_interval.iterrows()])


        img_fns = np.array([get_frame_path(args.base_path, row['speaker'], row['frame_fn']) for _, row in df_interval.iterrows()])

        interval_start = str(pd.to_datetime(df_interval.iloc[0]['pose_dt']).time())
        interval_end = str(pd.to_datetime(df_interval.iloc[-1]['pose_dt']).time())

        audio_out_path = AUDIO_FN_TEMPLATE % (speaker_name, interval, interval_start, interval_end)
        ###TODO:check how many missing audio files, save raw audio and proceesed audios from pats
        ###comapare raw audio and pats proceesed audios
        video_path = get_video_path(args.base_path, speaker_name, video_fn)
        if os.path.exists(video_path):
            save_audio_sample_from_video(video_path, audio_out_path, interval_start, interval_end)
            interval_audio_wav, sr = raw_repr(audio_out_path, SR)

        else:
            continue


        interval_start = pd.to_timedelta(df_interval.iloc[0]['pose_dt'])


        ### add audio from pats dataset
        pats_f = h5py.File(get_pats_path(args.pats_base_path, speaker_name, interval))
        audio_log_mel_400 = pats_f['audio/log_mel_400']

        if args.model == 'multimodal_context':
            text = []
            tokens = pats_f['text/tokens']
            for token in tokens:   
                word = tokenizer.decode([token])
                text.append(word)
        elif args.model == 'baseline' or args.model == 'Twostage':
            ### add text from pats dataset
            text = pats_f['text/bert']



        idx_400 = 0
        idx_512 = 0
        for idx in range(0, len(df_interval) - 64, 64):
            if idx_400+window_400 > audio_log_mel_400.shape[0]:
                
                continue###5713 in conan
            

            sample = df_interval[idx:idx + 64]
            start = (pd.to_timedelta(sample.iloc[0]['pose_dt'])-interval_start).total_seconds()*SR
            end = (pd.to_timedelta(sample.iloc[-1]['pose_dt'])-interval_start).total_seconds()*SR
            frames_out_path = TRAINING_SAMPLE_FN_TEMPLATE % (speaker_name, interval, sample.iloc[0]['pose_dt'], sample.iloc[-1]['pose_dt'])
            # wav = interval_audio_wav[int(start): int(end)]


            if not (os.path.exists(os.path.dirname(frames_out_path))):
                os.makedirs(os.path.dirname(frames_out_path))

            

            ###Audio and text slices
            pose = poses[idx:idx + 64]
            imgs = img_fns[idx:idx + 64]
            logmel = audio_log_mel_400[idx_400:idx_400+window_400]
            in_text = text[idx:idx + 64]


            idx_400 += window_hop_400




            logmel = np.expand_dims(logmel, 0)
            logmel = torch.tensor(logmel)
            logmel = logmel.float()
            logmel = logmel.to(device)

            if args.model == 'baseline' or args.model == 'Twostage':
                in_text = np.expand_dims(in_text, 0)
                in_text = torch.tensor(in_text)
                in_text = in_text.float()
                in_text = in_text.to(device)
            elif args.model == 'multimodal_context':
                in_text = np.array(in_text)
                in_text = np.expand_dims(in_text, 1)
        # wav = data['audio']
        # imgs = data['imgs']
        # if len(wav.shape) == 0 or wav.shape[0] == 0:
        #     continue

            keypoints2 = pose
            keypoints2 = keypoints2[:, pick_pose]

            if args.model == 'baseline':
                keypoints1 = generator(in_audio=logmel, in_text=in_text).cpu()
            elif args.model == 'multimodal_context':
                pre_seq = np.zeros((keypoints2.shape[0], keypoints2.shape[1] + 1))
                pre_seq = np.expand_dims(pre_seq, 0)
                pre_seq[:, 0:args.n_pre_poses, :-1] = keypoints2[0:args.n_pre_poses]
                pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
                pre_seq = torch.tensor(pre_seq)
                pre_seq = pre_seq.float()
                pre_seq = pre_seq.to(device)
                keypoints1, *_ = generator(pre_seq, in_audio=logmel, in_text=in_text)
                keypoints1 = keypoints1.cpu()
            elif args.model == 'Twostage':
                keypoints1, _, _ = generator(in_audio=logmel, in_text=in_text)
                keypoints1 = keypoints1.cpu()
            keypoints1 = keypoints1.detach().numpy()
            keypoints1_list.append(keypoints1[0])
            keypoints2_list.append(keypoints2)
        # wav_list.append(wav)
            imgs_list.append(imgs)


        # except Exception as e:
        #     # logger.exception(e)
        #     continue
        # df_loss.append(row_loss)
        

        keypoints1_list = np.array(keypoints1_list) 
        keypoints2_list = np.array(keypoints2_list)
        # print(len(wav_list))
        # wav_list = np.array(wav_list)

        imgs_list = np.array(imgs_list)
        wav_list.append(interval_audio_wav)
        wav_list = np.array(wav_list)

        name = args.name
        save_path = os.path.join('/mnt/lustressd/yuzhengming/visulization/conan', name, str(count), 'slices_npz_long/dev.npz')
        dir_path = os.path.join('/mnt/lustressd/yuzhengming/visulization/conan/', name, str(count), 'slices_npz_long')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        np.savez(save_path, keypoints1_list=keypoints1_list, keypoints2_list=keypoints2_list, wav_list=wav_list, imgs_list=imgs_list)
        
        count += 1
        if count >= 6:
            print('saving over')
            break
            

    # wav_list <- interval_audio_wav