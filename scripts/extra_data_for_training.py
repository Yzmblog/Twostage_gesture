##/mnt/lustressd/yuzhengming/data/S2G
##/mnt/lustressd/yuzhengming/gesture_data/pats/data/processed
import pandas as pd
import os
import numpy as np
import argparse
from logging import getLogger
import logging
from pandas.core.indexes import interval
from tqdm import tqdm
import h5py
import subprocess
from multiprocessing import Pool
import librosa

logging.basicConfig()


logger = getLogger(__name__)

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-b','--base_dataset_path', help="dataset root path")
parser.add_argument('-np','--num_processes', type=int, default=1)
parser.add_argument('-nf','--num_frames', type=int, default=64)
parser.add_argument('-s','--speaker', default=None)
parser.add_argument('-pats','--pats_base_path', default='/mnt/lustressd/yuzhengming/gesture_data/pats/data/processed')

args = parser.parse_args()

num_frames = args.num_frames
SR = 16000
AUDIO_FN_TEMPLATE = os.path.join(args.base_dataset_path, '%s/train/audio/%s-%s-%s.wav')
TRAINING_SAMPLE_FN_TEMPLATE = os.path.join(args.base_dataset_path, '%s/train/npz/%s-%s-%s.npz')
count = 0
samples_dropped = 0

def get_smpl_pose_path(base_path, speaker, pose_fn):
    new_pose_fn = pose_fn[:-4] + '.h5'
    return os.path.join(base_path, speaker, 'pose_smpl', new_pose_fn)


def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)


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


def raw_repr(path, sr=None):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def save_video_samples(df):
    global count
    global samples_dropped

    ##different fs 
    fs_512 = int(45.6*1000/512) #int(44.1*1000/512) #112 #round(22.5*1000/512)
    fs_400 = int(16.52 *1000/160)
    window_512 = int(4.3 * fs_512)
    window_400 = int(4.3 * fs_400)
    fs_ratio_512 = round(fs_512 / 15)
    fs_ratio_400 = round(fs_400 / 15)
    window_hop_512 = int(5*fs_ratio_512)
    window_hop_400 = int(5*fs_ratio_400)


    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'pose_fn': [], 'audio_fn': [], 'video_fn': [], 'speaker': []}
    intervals = df['interval_id'].unique()
    for interval in tqdm(intervals):
        try:
            df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
            video_fn = df_interval.iloc[0]['video_fn']
            speaker_name = df_interval.iloc[0]['speaker']
            if len(df_interval) < num_frames:
                logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
                continue

            ###TODO:change poses to smpl poses
            poses = np.array([h5py.File(get_smpl_pose_path(args.base_dataset_path, row['speaker'], row['pose_fn']), 'r')['poses'][:] for _, row in df_interval.iterrows()])


            ###################################
            #####################
            ###TODO: (it is neeed for indexing back to the real video frames)no need saving imgs and skip for conan  then save all speakers data
            #####################
            ###################################
            

            img_fns = np.array([get_frame_path(args.base_dataset_path, row['speaker'], row['frame_fn']) for _, row in df_interval.iterrows()])

            interval_start = str(pd.to_datetime(df_interval.iloc[0]['pose_dt']).time())
            interval_end = str(pd.to_datetime(df_interval.iloc[-1]['pose_dt']).time())

            audio_out_path = AUDIO_FN_TEMPLATE % (speaker_name, interval, interval_start, interval_end)
            ###TODO:check how many missing audio files, save raw audio and proceesed audios from pats
            ###comapare raw audio and pats proceesed audios
            video_path = get_video_path(args.base_dataset_path, speaker_name, video_fn)
            if os.path.exists(video_path):
                save_audio_sample_from_video(video_path, audio_out_path, interval_start, interval_end)
                interval_audio_wav, sr = raw_repr(audio_out_path, SR)

            else:
                interval_audio_wav = None
                count += 1
                print('missing raw audios number: %d' % count)##conan:195


            interval_start = pd.to_timedelta(df_interval.iloc[0]['pose_dt'])


            ### add audio from pats dataset
            pats_f = h5py.File(get_pats_path(args.pats_base_path, speaker_name, interval))
            audio_log_mel_400 = pats_f['audio/log_mel_400']
            # audio_log_mel_512 = pats_f['audio/log_mel_512']

            ### add text from pats dataset
            # text_w2v = pats_f['text/w2v']
            text_bert = pats_f['text/bert']
            # text_token = pats_f['text/tokens']
            
            idx_400 = 0
            idx_512 = 0
            for idx in range(0, len(df_interval) - num_frames, 5):
                if idx_400+window_400 > audio_log_mel_400.shape[0] or interval_audio_wav == None:##idx_512+window_512 > audio_log_mel_512.shape[0]:
                    samples_dropped += 1
                    print('drop this sample, already drop:%d' % (samples_dropped))
                    continue###5713 in conan
                

                sample = df_interval[idx:idx + num_frames]
                start = (pd.to_timedelta(sample.iloc[0]['pose_dt'])-interval_start).total_seconds()*SR
                end = (pd.to_timedelta(sample.iloc[-1]['pose_dt'])-interval_start).total_seconds()*SR
                frames_out_path = TRAINING_SAMPLE_FN_TEMPLATE % (speaker_name, interval, sample.iloc[0]['pose_dt'], sample.iloc[-1]['pose_dt'])
                wav = interval_audio_wav[int(start): int(end)] if interval_audio_wav is not None else 0


                if not (os.path.exists(os.path.dirname(frames_out_path))):
                    os.makedirs(os.path.dirname(frames_out_path))

                

                ###Audio and text slices
                #all
                # np.savez(frames_out_path, pose=poses[idx:idx + num_frames], imgs=img_fns[idx:idx + num_frames], audio=wav, 
                # audio_log_mel_400=audio_log_mel_400[idx_400:idx_400+window_400], audio_log_mel_512=audio_log_mel_512[idx_512:idx_512+window_512], text_w2v=text_w2v[idx:idx + num_frames], 
                # text_bert=text_bert[idx:idx + num_frames], text_token=text_token[idx:idx + num_frames])

                #part
                np.savez(frames_out_path, pose=poses[idx:idx + num_frames], imgs=img_fns[idx:idx + num_frames], audio=wav, 
                audio_log_mel_400=audio_log_mel_400[idx_400:idx_400+window_400],
                text_bert=text_bert[idx:idx + num_frames])

                data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
                data_dict["start"].append(sample.iloc[0]['pose_dt'])
                data_dict["end"].append(sample.iloc[-1]['pose_dt'])
                data_dict["interval_id"].append(interval)
                data_dict["pose_fn"].append(frames_out_path)
                data_dict["audio_fn"].append(audio_out_path)
                data_dict["video_fn"].append(video_fn)
                data_dict["speaker"].append(speaker_name)
                idx_400 += window_hop_400
                idx_512 += window_hop_512

            pats_f.close()
        except Exception as e:
            logger.exception(e)
            continue
    return pd.DataFrame.from_dict(data_dict)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(args.base_dataset_path, "frames_df_10_19_19.csv"))
    if args.speaker is not None:
        df = df[df['speaker'] == args.speaker]
    df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]


    intervals_unique = df['interval_id'].unique()
    print ("Number of unique intervals to save: %s" % (len(intervals_unique)))


    intervals = np.array_split(intervals_unique, args.num_processes)
    dfs = [df[df['interval_id'].isin(interval)] for interval in intervals]
    del df
    if args.num_processes > 1:
        p = Pool(args.num_processes)
        dfs = p.map(save_video_samples, dfs)
    else:
        dfs = map(save_video_samples, dfs)
    pd.concat(dfs).to_csv(os.path.join(args.base_dataset_path, "train_ellon.csv"), index=False)