# # import math
# # import numpy as np
# # import logging
# # import os
# # import pickle
# # import numpy as np
# # import fasttext
# # #print('{:010}'.format(2).encode('ascii'))
# # # import numpy as np
# # # from scipy.interpolate import interp1d
# # # x = range(0, 6)
# # # y = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
# # # f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
# # # x_new = np.arange(0, 6, 6 / 10)
# # # interpolated_y = f(x_new)
# # # print(interpolated_y)
# #
# # # print(math.floor(3.8))
# # # clip_audio = [[2, 3, 4], [1, 2, 3]]
# # # padded_data = np.pad(clip_audio, ((0, 0), (0, 5)), mode='symmetric')
# # # print(padded_data)
# # # pickle.load(open('./data/ted_dataset/vocab_cache.pkl', 'rb'))
# # # import random
# # # test = {'x': 1, 'y': 2, 'z': 3}
# # # vid_indices = [random.choice(list(test.values())) for _ in range(10)]
# # # print(vid_indices)
# #
# #
# # import datetime
# # import logging
# # import os
# # import pickle
# # import random
# #
# # import numpy as np
# # import lmdb as lmdb
# # import torch
# # from torch.nn.utils.rnn import pad_sequence
# #
# # from torch.utils.data import Dataset, DataLoader
# # from torch.utils.data.dataloader import default_collate
# #
# # import utils.train_utils
# # import utils.data_utils
# # from model.vocab import Vocab
# # from data_loader.data_preprocessor import DataPreprocessor
# # import pyarrow
# #
# #
# # def word_seq_collate_fn(data):
# #     """ collate function for loading word sequences in variable lengths """
# #     # sort a list by sequence length (descending order) to use pack_padded_sequence
# #     data.sort(key=lambda x: len(x[0]), reverse=True)
# #
# #     # separate source and target sequences
# #     word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)
# #
# #     # merge sequences
# #     words_lengths = torch.LongTensor([len(x) for x in word_seq])
# #     word_seq = pad_sequence(word_seq, batch_first=True).long()
# #
# #     text_padded = default_collate(text_padded)
# #     poses_seq = default_collate(poses_seq)
# #     vec_seq = default_collate(vec_seq)
# #     audio = default_collate(audio)
# #     spectrogram = default_collate(spectrogram)
# #     aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}
# #
# #     return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info
# #
# #
# # def default_collate_fn(data):
# #     _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)
# #
# #     text_padded = default_collate(text_padded)
# #     pose_seq = default_collate(pose_seq)
# #     vec_seq = default_collate(vec_seq)
# #     audio = default_collate(audio)
# #     spectrogram = default_collate(spectrogram)
# #     aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}
# #
# #     return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info
# #
# #
# # class SpeechMotionDataset(Dataset):
# #     def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec,
# #                  speaker_model=None, remove_word_timing=False):
# #
# #         self.lmdb_dir = lmdb_dir
# #         self.n_poses = n_poses
# #         self.subdivision_stride = subdivision_stride
# #         self.skeleton_resampling_fps = pose_resampling_fps
# #         self.mean_dir_vec = mean_dir_vec
# #         self.remove_word_timing = remove_word_timing
# #
# #         self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
# #         self.expected_spectrogram_length = utils.data_utils.calc_spectrogram_length_from_motion_length(
# #             n_poses, pose_resampling_fps)
# #
# #         self.lang_model = None
# #
# #         logging.info("Reading data '{}'...".format(lmdb_dir))
# #         preloaded_dir = lmdb_dir + '_cache'
# #         # if not os.path.exists(preloaded_dir):
# #         logging.info('Creating the dataset cache...')
# #         assert mean_dir_vec is not None
# #         if mean_dir_vec.shape[-1] != 3:
# #             mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
# #         n_poses_extended = int(round(n_poses * 1.25))  # some margin ### sample的时候加了margin
# #         data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses_extended,
# #                                         subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
# #         data_sampler.run()
# #         # else:
# #         #     logging.info('Found the cache {}'.format(preloaded_dir))
# #
# #         # init lmdb
# #         self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
# #         with self.lmdb_env.begin() as txn:
# #             self.n_samples = txn.stat()['entries']
# #
# #         # make a speaker model
# #         if speaker_model is None or speaker_model == 0:
# #             precomputed_model = lmdb_dir + '_speaker_model.pkl'
# #             if not os.path.exists(precomputed_model):
# #                 self._make_speaker_model(lmdb_dir, precomputed_model)
# #             else:
# #                 with open(precomputed_model, 'rb') as f:
# #                     self.speaker_model = pickle.load(f)
# #         else:
# #             self.speaker_model = speaker_model
# #
# #     def __len__(self):
# #         return self.n_samples
# #
# #     def __getitem__(self, idx):
# #         with self.lmdb_env.begin(write=False) as txn:
# #             key = '{:010}'.format(idx).encode('ascii')
# #             sample = txn.get(key)
# #
# #             sample = pyarrow.deserialize(sample)
# #             word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
# #
# #         def extend_word_seq(lang, words, end_time=None):
# #             n_frames = self.n_poses
# #             if end_time is None:
# #                 end_time = aux_info['end_time']
# #             frame_duration = (end_time - aux_info['start_time']) / n_frames
# #
# #             extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
# #             if self.remove_word_timing:
# #                 n_words = 0
# #                 for word in words:
# #                     idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
# #                     if idx < n_frames:
# #                         n_words += 1
# #                 space = int(n_frames / (n_words + 1))
# #                 for i in range(n_words):
# #                     idx = (i+1) * space
# #                     extended_word_indices[idx] = lang.get_word_index(words[i][0])
# #             else:
# #                 prev_idx = 0
# #                 for word in words:
# #                     idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
# #                     if idx < n_frames:
# #                         extended_word_indices[idx] = lang.get_word_index(word[0])
# #                         # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
# #                         prev_idx = idx
# #             return torch.Tensor(extended_word_indices).long()
# #
# #         def words_to_tensor(lang, words, end_time=None):
# #             indexes = [lang.SOS_token]
# #             for word in words:
# #                 if end_time is not None and word[1] > end_time:
# #                     break
# #                 indexes.append(lang.get_word_index(word[0]))
# #             indexes.append(lang.EOS_token)
# #             return torch.Tensor(indexes).long()
# #
# #         duration = aux_info['end_time'] - aux_info['start_time']
# #         do_clipping = True
# #
# #         if do_clipping:
# #             # print(self.n_poses, vec_seq.shape[0])
# #             sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
# #             audio = utils.data_utils.make_audio_fixed_length(audio, self.expected_audio_length)
# #             spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
# #             vec_seq = vec_seq[0:self.n_poses]
# #             pose_seq = pose_seq[0:self.n_poses]
# #         else:
# #             sample_end_time = None
# #
# #         # to tensors
# #         word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)## word->word index
# #         extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
# #         vec_seq = torch.from_numpy(vec_seq).reshape((vec_seq.shape[0], -1)).float()
# #         pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
# #         audio = torch.from_numpy(audio).float()
# #         spectrogram = torch.from_numpy(spectrogram)
# #
# #         return word_seq_tensor, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info
# #
# #     def set_lang_model(self, lang_model):
# #         self.lang_model = lang_model
# #
# #     def _make_speaker_model(self, lmdb_dir, cache_path):
# #         logging.info('  building a speaker model...')
# #         speaker_model = Vocab('vid', insert_default_tokens=False)
# #
# #         lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
# #         txn = lmdb_env.begin(write=False)
# #         cursor = txn.cursor()
# #         for key, value in cursor:
# #             video = pyarrow.deserialize(value)
# #             vid = video['vid']
# #             speaker_model.index_word(vid)
# #
# #         lmdb_env.close()
# #         logging.info('    indexed %d videos' % speaker_model.n_words)
# #         self.speaker_model = speaker_model
# #
# #         # cache
# #         with open(cache_path, 'wb') as f:
# #             pickle.dump(self.speaker_model, f)
# #
# #
# # mean_dir_vec = [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916]
# # mean_pose = [0.0000306,  0.0004946,  0.0008437,  0.0033759, -0.2051629, -0.0143453,  0.0031566, -0.3054764,  0.0411491,  0.0029072, -0.4254303, -0.001311 , -0.1458413, -0.1505532, -0.0138192, -0.2835603,  0.0670333,  0.0107002, -0.2280813,  0.112117 , 0.2087789,  0.1523502, -0.1521499, -0.0161503,  0.291909 , 0.0644232,  0.0040145,  0.2452035,  0.1115339,  0.2051307]
# #
# # mean_pose = np.array(mean_pose)
# # mean_dir_vec = np.array(mean_dir_vec)
# #
# #
# # test = SpeechMotionDataset('../data/ted_dataset/lmdb_train',
# #                                     n_poses=34,
# #                                     subdivision_stride=10,
# #                                     pose_resampling_fps=15,
# #                                     mean_dir_vec=mean_dir_vec,
# #                                     mean_pose=mean_pose,
# #                                     remove_word_timing=False
# #                                     )
# # test.__getitem__(0)
# #
# #26095
#
# # import numpy as np
# # test = np.load('/home/SENSETIME/yuzhengming/05jJodDVJRQ/00000.npz', allow_pickle=True)
# #
# # print(test['fname'])
#
#
# ###------csv file structure----------
# # 0 dataset
# # 1 delta_time
# # 2 end_time
# # 3 interval_id
# # 4 speaker
# # 5 start_time
# # 6 video_fn
# # 7 video_link
#
import os.path
import h5py
import numpy as np
import pandas as pd

import cv2
#
def capture_video(video_path, result_image_path, video, start_time, end_time, poses):
    """
    功能：截取短视频
    参数：
        video_path：需要截取的视频路径
        result_video_path：截取后的视频存放的路径
        video：需要截取的视频的名称（不带后缀）
        result_video：截取了的视频的名称（不带后缀）
        start_time：截取开始时间（单位s）
        end_time：截取结束时间（单位s）
    """

    # 读取视频
    path = os.path.join(video_path, video)
    cap = cv2.VideoCapture(path)

    # 读取视频帧率
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    stride = round(fps_video / 15)
    # print(stride)

    # 设置写入视频的编码格式
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 获取视频宽度和高度
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置写视频的对象
    # videoWriter = cv2.VideoWriter(result_video_path + result_video, fourcc, fps_video, (frame_width, frame_height))

    # 初始化一个计数器
    count = 0
    frameNum = 0
    while (cap.isOpened()):

        # 读取视屏里的图片
        ret, frame = cap.read()

        # 如果视屏没有读取结束
        if ret == True:

            # 计数器加一
            count += 1

            # 截取相应时间内的视频信息
            if(count >= (start_time * fps_video) and count <= (end_time * fps_video)):
                if round(count - (start_time * fps_video)) % stride == 0:
                    # 将图片写入视屏
                    result_path = os.path.join(result_image_path, str(count) + '.jpg')

                    if not os.path.exists(os.path.dirname(result_path)):
                        os.makedirs(os.path.dirname(result_path))
                    ## TODO: draw line in each frame
                    neckPoint = poses[frameNum][0]
                    poses[frameNum, 1:, ] += neckPoint
                    for i in range(1, 52):
                        pointOneX = int(poses[frameNum][i][0])
                        pointOneY = int(poses[frameNum][i][1])
                        pointTwoX = int(poses[frameNum][parent[i]][0])
                        pointTwoY = int(poses[frameNum][parent[i]][1])
                        cv2.line(frame, (pointOneX, pointOneY), (pointTwoX, pointTwoY), (0,0,255), 3)
                    cv2.imwrite(result_path, frame)
                    frameNum += 1

            if(count >= (end_time * fps_video)):
                break

        else:
            # 写入视屏结束
            # videoWriter.release()
            break
    print('frameNum2:', frameNum)
#
def StrTime2FloatTime(time):
    strTime = time[11:].split(':')
    t = 0
    for i, s in enumerate(strTime):
        t += float(s) * pow(60, 2 - i)
    return t
#
#
#
parent = [-1,
        0, 1, 2,
        0, 4, 5,
        0, 7, 7,
        6,
        10, 11, 12, 13,
        10, 15, 16, 17,
        10, 19, 20, 21,
        10, 23, 24, 25,
        10, 27, 28, 29,
        3,
        31, 32, 33, 34,
        31, 36, 37, 38,
        31, 40, 41, 42,
        31, 44, 45, 46,
        31, 48, 49, 50]

if __name__ == '__main__':


    ###---------old vis--------------------------------
    # df = pd.read_csv('/media/SENSETIME\yuzhengming/DATA/PATS-selected (1)/pats/data/cmu_intervals_df.csv')

    # video_name = df['video_fn']
    # video_stime = df['start_time']
    # video_etime = df['end_time']
    # inter_ids = df['interval_id']
    # speakers = df['speaker']
    # # deltaTimes = df['delta_time']

    # base_path = '/media/SENSETIME\yuzhengming/DATA/PATS/video/'
    # h5_path = '/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/'

    # length = 84289

    # for i in range(length):
    #     s_t = video_stime[i]
    #     e_t = video_etime[i]

    #     start_time = StrTime2FloatTime(s_t)
    #     end_time = StrTime2FloatTime(e_t)

    #     speaker = speakers[i]
    #     inter_id = inter_ids[i]
    #     video = video_name[i]


    #     if(speaker != 'ytch_prof'):

    #         continue

    #     f = h5py.File(os.path.join(h5_path, speaker, str(inter_id)+'.h5'), 'r')

    #     poses = np.array(f['/pose']['data'][()])
    #     poses = np.reshape(poses, newshape=(poses.shape[0], 2, -1)).transpose(0, 2, 1)
    #     print('framesNum1:', poses.shape[0])
    #     video_path = os.path.join(base_path, speaker, "videos")
    #     result_image_path = os.path.join(base_path, 'All_frames', speaker, inter_id)

    #     capture_video(video_path, result_image_path, video, start_time, end_time, poses)
    ###-----end old----------------


    ###--------visualize for test the pats poses data--------
    frame_path = '/media/SENSETIME\yuzhengming/DATA/PATS/crop_intervals/noah/cmu0000035915'
    h5_path = '/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/noah/cmu0000035915.h5'
    result_image_path = '/media/SENSETIME\yuzhengming/DATA/PATS/crop_intervals/noah/cmu0000035915/vis'

    f = h5py.File(h5_path)
    poses = np.array(f['/pose']['data'][()])
    poses = np.reshape(poses, newshape=(poses.shape[0], 2, -1)).transpose(0, 2, 1)
    all_frame = os.listdir(frame_path)
    frame_length = len(all_frame)
    poses_length = poses.shape[0]
    frames_draw = []


    for i in range(1, frame_length + 1, 2):
        single_fram = 'frame%d.jpg' % i
        frames_draw.append(single_fram)

    print(len(frames_draw), poses_length)
    # assert(len(frames_draw) == poses_length)
    
    for i, frame_name in enumerate(frames_draw):
        img_path = os.path.join(frame_path, frame_name)
        pose = poses[i]

        result_path = os.path.join(result_image_path, 'frame_vis_%d.jpg' % (i + 1))
        frame = cv2.imread(img_path)

        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        ## TODO: draw line in each frame
        neckPoint = pose[0]
        pose[1:, ] += neckPoint
        for j in range(1, 52):
            pointOneX = int(pose[j][0])
            pointOneY = int(pose[j][1])
            pointTwoX = int(pose[parent[j]][0])
            pointTwoY = int(pose[parent[j]][1])
            cv2.line(frame, (pointOneX, pointOneY), (pointTwoX, pointTwoY), (0,0,255), 3)
        cv2.imwrite(result_path, frame)


    ###-------end of this visualize-------------

    ###------h5 file structure-------
    # / audio-->['log_mel_400', 'log_mel_512', 'silence']
    # / pose-->['confidence', 'data', 'normalize']
    # / text-->['bert', 'meta', 'tokens', 'w2v']
    # text['meta']->['axis0', 'axis1', 'block0_items', 'block0_values', 'block1_items', 'block1_values']

    # f = h5py.File('/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/almaram/120149.h5', 'r')
    # poses = np.array(f['/pose']['data'][()])
    #
    #
    #
    # poses = np.reshape(poses, newshape=(poses.shape[0], 2, -1)).transpose(0, 2, 1)
    # print(poses)
    # neckPoint = poses[0][0]
    # # print(neckPoint)
    # # print(poses)
    # poses[:, 1:, ] += neckPoint
    # print(poses)
    # 打印出文件中的关键字
    #     print(f[key].shape)
    # 将key换成某个文件中的关键字,打印出某个数据的大小尺寸
    #     print(f[key][:])
    # 将key换成某个文件中的关键字,打印出该数据(数组)


    # print(len(parent))
# # H:360/720 W:480/1280

# import torch
# a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# b = a.reshape((a.shape[0], -1))
# print(b, b.reshape((b.shape[0], -1, 2)))
# import torch.nn

# rnn = torch.nn.RNN()
