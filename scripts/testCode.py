# test = set(("alarm"))
# test.add("alarm")
# print(test)
# test = dict()
from re import T
import torch
# import numpy as np
# # x = torch.tensor(1)
# # test[x] = 1
# # print(test)
# t = [[6, 0, 1.9593],
#      [3,  4,  1.6982],
#      [1, 6,  0.7860]]
# x = np.array(t)
# a = np.min(x, axis=0)
# b = np.max(x, axis=0)
# h = b - a
# print(b, a, h, np.linalg.norm(h))
# path = './data/xxx.pkl'
# print(path.split("/")[-1])
import os
import pandas as pd
from tqdm import tqdm
import json
import subprocess
# import librosa
# df = pd.read_csv('/media/SENSETIME\yuzhengming/DATA/S2G/frames_df_10_19_19.csv')
# df = pd.read_csv('/mnt/lustressd/yuzhengming/data/S2G/frames_df_10_19_19.csv')

# intervals = df['interval_id'].unique()

# missing = dict()
# missing['frame'] = []
# for interval in tqdm(intervals):
#     df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)

#     for _, row in df_interval.iterrows():
#         if (row['speaker'] != 'shelly'):
#             continue
#         # if row['pose_fn'][:-4] != row['frame_fn'][2:-4]:
#         #     print('frame name:', row['frame_fn'], 
#         #     'pose_name:', row['pose_fn'])
#         if not os.path.exists(os.path.join('/mnt/lustressd/yuzhengming/data/S2G/shelly/keypoints_simple', row['pose_fn'])):
#             print(row['frame_fn'])
#             missing['frame'].append(row['frame_fn'])

#     with open("/mnt/lustressd/yuzhengming/data/S2G/missing_record.json","w") as f:
#         json.dump(missing,f)    
    
    # speaker_name = df_interval.iloc[0]['speaker']
    # if speaker_name != 'almaram':
    #     continue

# x = [272.438,104.187,0.853027,286.154,152.147,0.772148,245.055,150.175,0.631479,206.898,229.37,0.739349,208.849,182.441,0.50854,331.121,153.125,0.674568,347.777,232.322,0.707721,316.45,174.599,0.679799,287.124,321.357,0.401606,257.759,322.321,0.38114,0,0,0,0,0,0,318.41,322.328,0.367183,0,0,0,0,0,0,263.658,92.453,0.870111,286.109,93.4372,0.868505,254.842,99.2899,0.163686,307.634,99.3059,0.856891,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# #"face_keypoints_2d":
# [250.741,92.8385,0.510121,250.514,98.7444,0.634057,250.287,105.559,0.613324,250.514,112.6,0.596817,250.514,119.642,0.535128,252.331,126.684,0.548234,256.647,131.681,0.576424,263.461,135.088,0.685437,270.957,136.678,0.647345,278.68,136.905,0.64091,285.494,134.861,0.63864,292.082,129.636,0.611077,296.17,123.958,0.571599,299.805,117.371,0.586678,302.076,111.238,0.63103,303.439,103.969,0.681399,303.439,96.0186,0.583089,255.738,85.3427,0.679591,258.237,83.7526,0.773937,261.871,83.7526,0.804346,265.733,84.4341,0.79167,269.14,86.0241,0.759778,280.043,86.0241,0.736012,284.586,85.5698,0.799471,288.674,85.797,0.89002,292.763,86.9327,0.773183,295.943,89.4313,0.738187,273.683,92.6114,0.818135,272.547,96.7001,0.9205,272.093,100.789,0.846456,270.957,105.559,0.867199,266.868,107.83,0.848741,268.686,109.42,0.875404,270.957,110.556,0.821593,274.137,110.329,0.704046,276.409,109.42,0.536107,260.508,93.9743,0.721938,262.325,93.0657,0.730245,265.051,93.0657,0.755078,267.095,93.2928,0.761267,265.051,94.2014,0.760064,262.553,94.6557,0.760893,283.223,94.6557,0.77697,285.267,93.2928,0.821016,288.447,93.7471,0.758202,290.492,94.8829,0.745627,287.993,96.2458,0.745705,285.04,95.3372,0.833804,262.098,115.553,0.762468,266.641,114.645,0.760995,269.594,114.872,0.780535,272.093,115.78,0.785383,275.954,115.553,0.755627,280.497,117.143,0.650947,285.267,120.551,0.474735,281.86,124.185,0.78048,276.409,126.002,0.744663,271.866,125.321,0.727245,267.777,123.958,0.764009,264.597,120.551,0.745263,264.824,116.689,0.668139,269.14,117.598,0.799705,272.32,118.506,0.831338,276.181,118.733,0.727863,283.223,120.778,0.524258,276.181,120.323,0.723562,272.32,120.096,0.844989,269.14,118.733,0.795359,263.461,93.2928,0.767055,286.857,94.6557,0.803139],
# #"hand_left_keypoints_2d":
# [311.754,168.613,0.246088,311.754,158.222,0.352135,309.675,146.94,0.213898,299.878,132.393,0.062219,295.722,129.721,0.116125,309.972,140.706,0.47167,307.003,128.83,0.667093,297.8,127.94,0.58978,293.05,128.83,0.429855,307.597,143.081,0.536249,302.55,132.987,0.561873,291.268,130.909,0.401627,285.627,132.096,0.373873,303.144,145.753,0.459493,298.097,138.331,0.54171,287.112,135.956,0.367863,282.955,135.362,0.306614,298.394,149.909,0.652257,290.971,144.862,0.515661,285.924,144.268,0.399604,281.174,142.487,0.452895],
# #"hand_right_keypoints_2d":
# [214.975,166.954,0.0409515,200.141,173.404,0.0624639,184.339,173.404,0.0330189,208.526,132.448,0.0198233,208.203,131.481,0.0304888,209.493,132.126,0.0653879,177.567,186.626,0.0481195,175.309,187.271,0.0492295,172.084,185.659,0.0226036,189.821,177.596,0.0238992,184.984,185.336,0.0845353,182.727,193.721,0.0512563,180.147,183.724,0.0303971,202.398,175.661,0.0243802,191.434,183.724,0.0723525,192.079,186.626,0.0912014,194.336,200.816,0.0271791,204.978,175.661,0.0725978,204.011,183.079,0.120625,204.011,186.626,0.16486,203.043,188.238,0.136706]

# x = [272.438,104.187,0.853027,286.154,152.147,0.772148,245.055,150.175,0.631479,206.898,229.37,0.739349,208.849,182.441,0.50854,331.121,153.125,0.674568,347.777,232.322,0.707721,316.45,174.599,0.679799,287.124,321.357,0.401606,257.759,322.321,0.38114,0,0,0,0,0,0,318.41,322.328,0.367183,0,0,0,0,0,0,263.658,92.453,0.870111,286.109,93.4372,0.868505,254.842,99.2899,0.163686,307.634,99.3059,0.856891,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# #"face_keypoints_2d":
# y = [250.741,92.8385,0.510121,250.514,98.7444,0.634057,250.287,105.559,0.613324,250.514,112.6,0.596817,250.514,119.642,0.535128,252.331,126.684,0.548234,256.647,131.681,0.576424,263.461,135.088,0.685437,270.957,136.678,0.647345,278.68,136.905,0.64091,285.494,134.861,0.63864,292.082,129.636,0.611077,296.17,123.958,0.571599,299.805,117.371,0.586678,302.076,111.238,0.63103,303.439,103.969,0.681399,303.439,96.0186,0.583089,255.738,85.3427,0.679591,258.237,83.7526,0.773937,261.871,83.7526,0.804346,265.733,84.4341,0.79167,269.14,86.0241,0.759778,280.043,86.0241,0.736012,284.586,85.5698,0.799471,288.674,85.797,0.89002,292.763,86.9327,0.773183,295.943,89.4313,0.738187,273.683,92.6114,0.818135,272.547,96.7001,0.9205,272.093,100.789,0.846456,270.957,105.559,0.867199,266.868,107.83,0.848741,268.686,109.42,0.875404,270.957,110.556,0.821593,274.137,110.329,0.704046,276.409,109.42,0.536107,260.508,93.9743,0.721938,262.325,93.0657,0.730245,265.051,93.0657,0.755078,267.095,93.2928,0.761267,265.051,94.2014,0.760064,262.553,94.6557,0.760893,283.223,94.6557,0.77697,285.267,93.2928,0.821016,288.447,93.7471,0.758202,290.492,94.8829,0.745627,287.993,96.2458,0.745705,285.04,95.3372,0.833804,262.098,115.553,0.762468,266.641,114.645,0.760995,269.594,114.872,0.780535,272.093,115.78,0.785383,275.954,115.553,0.755627,280.497,117.143,0.650947,285.267,120.551,0.474735,281.86,124.185,0.78048,276.409,126.002,0.744663,271.866,125.321,0.727245,267.777,123.958,0.764009,264.597,120.551,0.745263,264.824,116.689,0.668139,269.14,117.598,0.799705,272.32,118.506,0.831338,276.181,118.733,0.727863,283.223,120.778,0.524258,276.181,120.323,0.723562,272.32,120.096,0.844989,269.14,118.733,0.795359,263.461,93.2928,0.767055,286.857,94.6557,0.803139]
# #"hand_left_keypoints_2d":
# z = [311.754,168.613,0.246088,311.754,158.222,0.352135,309.675,146.94,0.213898,299.878,132.393,0.062219,295.722,129.721,0.116125,309.972,140.706,0.47167,307.003,128.83,0.667093,297.8,127.94,0.58978,293.05,128.83,0.429855,307.597,143.081,0.536249,302.55,132.987,0.561873,291.268,130.909,0.401627,285.627,132.096,0.373873,303.144,145.753,0.459493,298.097,138.331,0.54171,287.112,135.956,0.367863,282.955,135.362,0.306614,298.394,149.909,0.652257,290.971,144.862,0.515661,285.924,144.268,0.399604,281.174,142.487,0.452895]
# #"hand_right_keypoints_2d":
# n = [214.975,166.954,0.0409515,200.141,173.404,0.0624639,184.339,173.404,0.0330189,208.526,132.448,0.0198233,208.203,131.481,0.0304888,209.493,132.126,0.0653879,177.567,186.626,0.0481195,175.309,187.271,0.0492295,172.084,185.659,0.0226036,189.821,177.596,0.0238992,184.984,185.336,0.0845353,182.727,193.721,0.0512563,180.147,183.724,0.0303971,202.398,175.661,0.0243802,191.434,183.724,0.0723525,192.079,186.626,0.0912014,194.336,200.816,0.0271791,204.978,175.661,0.0725978,204.011,183.079,0.120625,204.011,186.626,0.16486,203.043,188.238,0.136706]

# print(len(x), len(y), len(z), len(n))

# with open ('/media/SENSETIME\yuzhengming/DATA/S2G/missing_record.json') as f:
#     imgs = json.load(f)
#     for frame_name in imgs['frame']:
#         img_path = os.path.join('/media/SENSETIME\\\\yuzhengming/DATA/shelly/frames', frame_name)
#         cmd = 'cp %s %s' % (img_path, '/home/SENSETIME/yuzhengming/Projects/openpose-1.6.0/examples/media/shelly')
#         subprocess.call(cmd, shell=True)

###-----------use cv2 to save every frame------------------
# import cv2
# import os,sys
# cap = cv2.VideoCapture('/media/SENSETIME\yuzhengming/DATA/PATS/crop_intervals/noah/cmu0000035915_wEaAUZfe8bY_00:09:11.466666-00:10:02.mp4')
# isOpened = cap.isOpened
# print(isOpened)

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(fps,width,height)

# base_path = '/media/SENSETIME\yuzhengming/DATA/PATS/crop_intervals/noah/cmu0000035915/cv2/'

# # print(os.getcwd())
# flag = True
# i = 0
# while(flag):
#     # if i == 100:
#     #     break
#     # else:
#     i = i+1
#     (flag,frame) = cap.read()   #读取每一帧，flag表示是否读取成功，frame为图片内容。
#     if not flag:
#         break
#     fileName = base_path + "image" +str(i) +".jpg"
#     print(fileName)

#     cv2.imwrite(fileName,frame)

# # os.chdir(r"C:\Users\lenovo\Desktop\python\jiqixuexi")
# # print(os.getcwd())
# cap.release()

# print("end!")
###-----------------------------end of using cv2------------------------------------------

# import numpy as np
# import h5py

# # h5_path = '/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/almaram/120145.h5'
# # f = h5py.File(h5_path)
# # poses = np.array(f['/pose']['data'][()])
# # poses = np.reshape(poses, newshape=(poses.shape[0], 2, -1)).transpose(0, 2, 1)

# poses = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [10, 11]]])
# print(poses)

# neckPoint = poses[:, [0]]
# print(neckPoint)
# poses[:, 1:, ] += neckPoint
# print(poses)

# print(poses.shape)
# avg_pose = np.mean(poses, axis=0)
# print(avg_pose.shape)
# print(avg_pose)

SR = 16000

def save_audio_sample_from_video(vid_path, audio_out_path, audio_start, audio_end, sr=44100):
    if not (os.path.exists(os.path.dirname(audio_out_path))):
        os.makedirs(os.path.dirname(audio_out_path))
    cmd = 'ffmpeg -i "%s" -ss %s -to %s -ab 160k -ac 2 -ar %s -vn "%s" -y -loglevel warning' % (
    vid_path, audio_start, audio_end, sr, audio_out_path)
    subprocess.call(cmd, shell=True)

# def raw_repr(path, sr=None):
#     wav, sr = librosa.load(path, sr=sr, mono=True)
#     return wav, sr

import numpy as np

if __name__ == "__main__":
    speaker = 'conan'
    # df = pd.read_csv('/mnt/lustressd/yuzhengming/data/S2G/frames_df_10_19_19.csv')
    df = pd.read_csv('/media/SENSETIME\yuzhengming/DATA/S2G/frames_df_10_19_19.csv')

    df = df[df['speaker'] == speaker]

    intervals_unique = df['interval_id'].unique()
    print("number of unique intervals %s" % (len(intervals_unique)))

    intervals = np.array_split(intervals_unique, 1)

    dfs = [df[df['interval_id'].isin(interval)] for interval in intervals]
    del df

    dfs = dfs[0]
    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'pose_fn': [], 'audio_fn': [], 'video_fn': [], 'speaker': []}
    intervals = dfs['interval_id'].unique()



    for interval in tqdm(intervals):
        df_interval = dfs[dfs['interval_id'] == interval].sort_values('frame_id', ascending=True)
        video_fn = df_interval.iloc[0]['video_fn']
        speaker_name = df_interval.iloc[0]['speaker']
        if not (video_fn == 'Monologue_06_11_12_-_CONAN_on_TBS-gWvqxKrO0PM.mkv'):
            continue
        # print('yes')
        # if video_fn != '':
        #     continue

        # if len(df_interval) < num_frames:
        #     logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
        #     continue
        
        img_fns = []
        smpl_fns = []
        for _, row in df_interval.iterrows():
            img_fns.append(os.path.join('/media/SENSETIME\\\\yuzhengming/DATA/S2G/conan/frames', row['frame_fn']))

            temp = row['frame_fn'].split('.')[:-1]
            out = temp[0] + '.' + temp[1] + '.h5'
            smpl_fns.append(os.path.join('/media/SENSETIME\\\\yuzhengming/DATA/S2G/conan/smpl',  out))

        out_path_img = '/home/SENSETIME/yuzhengming/Digital_human/data/conan/frames'
        out_path_img = os.path.join(out_path_img, str(interval))
        out_path_smpl = '/home/SENSETIME/yuzhengming/Digital_human/data/conan/smpl'
        out_path_smpl = os.path.join(out_path_smpl, str(interval))

        if not os.path.exists(out_path_img):
            os.makedirs(out_path_img)
        if not os.path.exists(out_path_smpl):
            os.makedirs(out_path_smpl)

        for i in range(len(img_fns)):
            img_fn = img_fns[i]
            smpl_fn = smpl_fns[i]
            img_out = os.path.join(out_path_img, '%d.jpg' % i)
            cmd1 = 'cp ' + img_fn + ' ' + img_out
            subprocess.call(cmd1, shell=True)

            # print(smpl_fn)
            smpl_out = os.path.join(out_path_smpl, '%d.h5' % i)
            cmd2 = 'cp ' + smpl_fn + ' ' + smpl_out
            subprocess.call(cmd2, shell=True)
        # interval_start = str(pd.to_datetime(df_interval.iloc[0]['pose_dt']).time())
        # interval_end = str(pd.to_datetime(df_interval.iloc[-1]['pose_dt']).time())

        # # audio_out_path = AUDIO_FN_TEMPLATE % (speaker_name, interval, interval_start, interval_end)
        # save_audio_sample_from_video(get_video_path(args.base_dataset_path, speaker_name, video_fn), audio_out_path, interval_start, interval_end)

        # interval_start = pd.to_timedelta(df_interval.iloc[0]['pose_dt'])

        # interval_audio_wav, sr = raw_repr(audio_out_path, SR)
        # # for idx in range(0, len(df_interval) - num_frames, 5):

        # sample = df_interval
        # start = (pd.to_timedelta(sample.iloc[0]['pose_dt'])-interval_start).total_seconds()*SR
        # end = (pd.to_timedelta(sample.iloc[-1]['pose_dt'])-interval_start).total_seconds()*SR
        # frames_out_path = TRAINING_SAMPLE_FN_TEMPLATE % (speaker_name, interval, sample.iloc[0]['pose_dt'], sample.iloc[-1]['pose_dt'])
        # wav = interval_audio_wav[int(start): int(end)]

        # if not (os.path.exists(os.path.dirname(frames_out_path))):
        #     os.makedirs(os.path.dirname(frames_out_path))

        # np.savez(frames_out_path, pose=poses[idx:idx + num_frames], imgs=img_fns[idx:idx + num_frames], audio=wav)

        # data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
        # data_dict["start"].append(sample.iloc[0]['pose_dt'])
        # data_dict["end"].append(sample.iloc[-1]['pose_dt'])
        # data_dict["interval_id"].append(interval)
        # data_dict["pose_fn"].append(frames_out_path)
        # data_dict["audio_fn"].append(audio_out_path)
        # data_dict["video_fn"].append(video_fn)
        # data_dict["speaker"].append(speaker_name)