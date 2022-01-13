
import sys
[sys.path.append(i) for i in ['.', '..']] ### Add search path of 'import'

from re import A
from matplotlib.pyplot import axis
import numpy as np
import torch
from model.Audio2Gesture import A2GNet
from transformers import BertTokenizer
import h5py
from model.Audio2Gesture import A2GNet
# from config.parse_args import parse_args
import torch.nn.functional as F
from config.parse_args import parse_args

# skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'), (1, 5, 'darkgreen'),
#                        (5, 6, 'limegreen'), (6, 7, 'darkseagreen')]
# dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
#                  (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length




# def convert_dir_vec_to_pose(vec):
#     vec = np.array(vec)
#
#     if vec.shape[-1] != 3:
#         vec = vec.reshape(vec.shape[:-1] + (-1, 3))
#
#     if len(vec.shape) == 2:
#         joint_pos = np.zeros((10, 3))
#         for j, pair in enumerate(dir_vec_pairs):
#             joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
#     elif len(vec.shape) == 3:
#         joint_pos = np.zeros((vec.shape[0], 10, 3))
#         for j, pair in enumerate(dir_vec_pairs):
#             joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
#     elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
#         joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
#         for j, pair in enumerate(dir_vec_pairs):
#             joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
#     else:
#         assert False
#
#     return joint_pos


# def convert_dir_to_rotation(dir):
#     rot = np.zeros((dir.shape[0], dir.shape[1], len(dir_vec_pairs)))


# args = parse_args()
# generator = A2GNet(args)
#
# audioV = torch.randn((4, 64, 128))
# sample = [torch.randn((4, 64, 16)).transpose(1, 2) for _ in range(2)]
# pose = torch.randn((4, 64, 104))
# target = torch.randn((4, 64, 104))



# print(total_loss, loss_share_align, loss_motion1, loss_motion2, loss_relax, loss_cyc, loss_div)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# print(tokenizer('hello, i am a test.'))##->input_ids(token id),  token_type_ids:0/1:第一句或第二句
import pandas as pd
###------h5 file structure-------
# / audio-->['log_mel_400', 'log_mel_512', 'silence']
# / pose-->['confidence', 'data', 'normalize']
# / text-->['bert', 'meta', 'tokens', 'w2v']'w2v':word to vector
# text['meta']->['axis0', 'axis1', 'block0_items', 'block0_values', 'block1_items', 'block1_values']
# text_df = pd.read_hdf('/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/almaram/120149.h5', key='text/meta')
# f = h5py.File('/media/SENSETIME\yuzhengming/DATA/PATS/all/data/processed/almaram/120149.h5', 'r')
# #
# print(np.array(f['/audio']['silence']))
# print(text_df[(1 <= text_df['end_frame']) & (60 > text_df['start_frame'])].Word.values)
from data_loader import Data
from train import evaluate_sample_and_save_video
from config.parse_args import parse_args
# from tqdm import tqdm
# #
#
# parents = [-1,
#             0, 1, 2,
#             0, 4, 5,
#             6,
#             10, 11, 12, 13,
#             10, 15, 16, 17,
#             10, 19, 20, 21,
#             10, 23, 24, 25,
#             10, 27, 28, 29,
#             3,
#             31, 32, 33, 34,
#             31, 36, 37, 38,
#             31, 40, 41, 42,
#             31, 44, 45, 46,
#             31, 48, 49, 50]
#
# joints_name = ['Neck',
#             'RShoulder', 'RElbow', 'RWrist',
#             'LShoulder', 'LElbow', 'LWrist',
#             'LHandRoot',
#             'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
#             'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
#             'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
#             'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
#             'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
#             'RHandRoot',
#             'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
#             'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
#             'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
#             'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
#             'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
#             ]
#
# dir_vec_pairs = []
#
# for i in range(len(parents)):
#     if parents[i] > 6:
#         parents[i] = parents[i] - 3
# # print(parents)
#		#-gencode arch=compute_20,code=sm_21 \
# for i in range(1, len(parents)):
#     dir_vec_pairs.append((parents[i], i, '{first_joint} To {second_joint}'.format(first_joint=joints_name[parents[i]],
#     second_joint=joints_name[i])))
#


# print(dir_vec_pairs)

# average_length = torch.zeros((48))

#
common_kwargs = dict(path2data='/media/SENSETIME\yuzhengming/DATA/PATS/all/data',
                     speaker=['chemistry'],
                     modalities=['pose/data', 'audio/log_mel_512', 'text/tokens'],
                     fs_new=[15, 15, 15],
                     batch_size=2,
                     window_hop=5,
                     load_data=False)

data = Data(**common_kwargs)
#
# # test = {'a':1, 'b':2}
# # print(test.items())
# # all_data = [data.train, data.dev, data.test]
#
# # length_avg_list = []
#
# # for specific_data in all_data:
# result = []
# with open('/home/SENSETIME/yuzhengming/Projects/baseline/bert-base-uncased-vocab.txt', mode='r') as tokens:
#     for x in tokens:
#         result.append(list(x.strip('\n').split(',')))
# # print(result)
# for i, batch in enumerate(data.train):
#
#     ### one batch
#     # pose = batch['text/data'] ## B * N * 49 * 2
#     text = batch['text/tokens'] ## B * N * 49 * 2
#     print(text.shape)
#         temp_length = torch.zeros((1, 64, 48, 2))
#         for i, pair in enumerate(dir_vec_pairs):
#             temp_length[:, :, i, ] = pose[:, :, pair[1],] - pose[:, :, pair[0],]

#         length = torch.sqrt(pow(temp_length[:, :, :, 0], 2) + pow(temp_length[:, :, :, 1], 2))
#         # print(length[0][0])
#         avg_length = torch.mean(length, dim=0)
#         # print(avg_length.shape)
#         avg_length = torch.mean(avg_length, dim = 0)
#         # print(avg_length)
#         length_avg_list.append(list(avg_length))

# length_avg_list = torch.tensor(length_avg_list)
# final_avg = torch.mean(length_avg_list, dim=0)
# print(final_avg)


# avg_length = [127.7295, 182.5979, 105.9161, 128.6834, 187.8634, 111.6433,  30.3583,
#          28.0698,  29.6881,  23.6946,  20.1090,  49.4278,  24.6465,  16.9815,
#          15.1073,  44.1553,  26.3268,  18.3899,  14.8320,  42.9849,  24.7423,
#          17.1161,  13.7727,  45.5155,  21.0095,  14.0150,  11.3402,  23.5129,
#          26.6342,  28.5038,  21.9957,  19.5356,  48.6835,  23.7355,  16.3687,
#          14.4559,  43.0570,  25.5599,  17.8005,  13.9920,  41.3955,  24.0277,
#          16.3925,  13.0776,  43.4481,  19.8831,  13.1203,  10.7297]
#
# test_data_loader = data.dev
# #
device = "cuda:0" if torch.cuda.is_available() else "cpu"

args = parse_args()
checkpoint = torch.load('/home/SENSETIME/yuzhengming/Downloads/onlymotion/Audio2Gesture_checkpoint_010.bin')

generator = A2GNet(args).to(device)

generator.load_state_dict(checkpoint['gen_dict'])

# for k in generator:
#     print(k)
# args = parse_args()

test_data_loader = data.dev
evaluate_sample_and_save_video(
    checkpoint['epoch'], args.name, test_data_loader, generator,
    args=args, save_path='/home/SENSETIME/yuzhengming/Projects/video_output/motion_effect_test/only_motion')

    ##'audio/log_mel_512' 'pose/data'


# for iter_idx, data in enumerate(data.dev, 0):
#     print(iter_idx, data)


# for i, batch in enumerate(data.train):
#     if i == 0:
#         print(batch['pose/normalize'])
#     else:
#         break


# for batch in data.train:
#
#     for key in batch.keys():
#       if key != 'meta':
#         print('{}: {}'.format(key, batch[key].shape))



# print(len(parent))
# H:360/720 W:480/1280
# from pathlib import Path
# # print(Path('/media/SENSETIME\yuzhengming/DATA/PATS/all/data').parent)
# # print(intervals)
# from sklearn.preprocessing import normalize
# import torch
# dir_vec = torch.tensor([[[1, 2]], [[3, 4]]])
# print(normalize(dir_vec[:, 0, :], axis=1))
# pose = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
# pose = pose.reshape(pose.shape[0], 2, -1)
# print(pose)
# sub = [0]
# print(pose.transpose(2, 1)[:, sub, ])

# avg_length = [127.7295, 182.5979, 105.9161, 128.6834, 187.8634, 111.6433,  30.3583,
#          28.0698,  29.6881,  23.6946,  20.1090,  49.4278,  24.6465,  16.9815,
#          15.1073,  44.1553,  26.3268,  18.3899,  14.8320,  42.9849,  24.7423,
#          17.1161,  13.7727,  45.5155,  21.0095,  14.0150,  11.3402,  23.5129,
#          26.6342,  28.5038,  21.9957,  19.5356,  48.6835,  23.7355,  16.3687,
#          14.4559,  43.0570,  25.5599,  17.8005,  13.9920,  41.3955,  24.0277,
#          16.3925,  13.0776,  43.4481,  19.8831,  13.1203,  10.7297]
#
#
# test = [(0, 1, 'Neck To RShoulder'), (1, 2, 'RShoulder To RElbow'), (2, 3, 'RElbow To RWrist'), (0, 4, 'Neck To LShoulder'),
#  (4, 5, 'LShoulder To LElbow'), (5, 6, 'LElbow To LWrist'), (6, 7, 'LWrist To LHandRoot'), (7, 8, 'LHandRoot To LHandThumb1'),
#  (8, 9, 'LHandThumb1 To LHandThumb2'), (9, 10, 'LHandThumb2 To LHandThumb3'), (10, 11, 'LHandThumb3 To LHandThumb4'),
#  (7, 12, 'LHandRoot To LHandIndex1'), (12, 13, 'LHandIndex1 To LHandIndex2'), (13, 14, 'LHandIndex2 To LHandIndex3'), (14, 15, 'LHandIndex3 To LHandIndex4'), (7, 16, 'LHandRoot To LHandMiddle1'), (16, 17, 'LHandMiddle1 To LHandMiddle2'), (17, 18, 'LHandMiddle2 To LHandMiddle3'), (18, 19, 'LHandMiddle3 To LHandMiddle4'), (7, 20, 'LHandRoot To LHandRing1'), (20, 21, 'LHandRing1 To LHandRing2'), (21, 22, 'LHandRing2 To LHandRing3'), (22, 23, 'LHandRing3 To LHandRing4'), (7, 24, 'LHandRoot To LHandLittle1'), (24, 25, 'LHandLittle1 To LHandLittle2'), (25, 26, 'LHandLittle2 To LHandLittle3'), (26, 27, 'LHandLittle3 To LHandLittle4'), (3, 28, 'RWrist To RHandRoot'), (28, 29, 'RHandRoot To RHandThumb1'), (29, 30, 'RHandThumb1 To RHandThumb2'), (30, 31, 'RHandThumb2 To RHandThumb3'), (31, 32, 'RHandThumb3 To RHandThumb4'), (28, 33, 'RHandRoot To RHandIndex1'), (33, 34, 'RHandIndex1 To RHandIndex2'), (34, 35, 'RHandIndex2 To RHandIndex3'), (35, 36, 'RHandIndex3 To RHandIndex4'), (28, 37, 'RHandRoot To RHandMiddle1'), (37, 38, 'RHandMiddle1 To RHandMiddle2'), (38, 39, 'RHandMiddle2 To RHandMiddle3'), (39, 40, 'RHandMiddle3 To RHandMiddle4'), (28, 41, 'RHandRoot To RHandRing1'), (41, 42, 'RHandRing1 To RHandRing2'), (42, 43, 'RHandRing2 To RHandRing3'), (43, 44, 'RHandRing3 To RHandRing4'), (28, 45, 'RHandRoot To RHandLittle1'), (45, 46, 'RHandLittle1 To RHandLittle2'), (46, 47, 'RHandLittle2 To RHandLittle3'), (47, 48, 'RHandLittle3 To RHandLittle4')]
#
# final = []
# for i, pair in enumerate(test):
#     final.append((pair[0], pair[1], round(avg_length[i] / 100, 2), pair[2]))
# print(final)
# import torch.nn as nn
# from torch.nn.utils import weight_norm

# # conv1 = weight_norm(nn.Conv1d(6, 8, 2, stride=1, padding=1, dilation=1))
# gru = nn.GRU(6, hidden_size=8, num_layers=3, batch_first=True,
#                   bidirectional=True, dropout=0.2)
# test = torch.randn((10, 64, 6))
# out, hidden = gru(test)
# print(out.shape, hidden.shape)
