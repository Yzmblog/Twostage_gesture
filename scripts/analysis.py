from random import random
import subprocess
import sys
from PIL.Image import NONE, new
from matplotlib import colors
from pandas.core import frame

from torch import median, ne
[sys.path.append(i) for i in ['.', '..']] ### Add search path of 'import'

import pandas as pd
import os
# from config.parse_args import parse_args
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
import argparse
import logging
from logging import getLogger
import json
import warnings
import torch
from transformers import BertTokenizer
from matplotlib.pyplot import MultipleLocator, axis


warnings.filterwarnings('ignore')

logging.basicConfig()
logger = getLogger(__name__)

##joints_list = ['Pelvis'(overall)0, 'L_Hip'(left datui)1, 'R_Hip'2, 'Spine1'(jizhu/yao xia)3, 'L_Knee'(left xi gai)4, 'R_Knee'5,
#'Spine2'(yao shang du zi shang)6, 'L_Ankle'7, 'R_Ankle'8, 'Spine3'(jizhu zuishang)9, 'L_Foot'(qian jiao zhang)10, 'R_Foot'11,
#'Neck'12, 'L_Collar'(left da bi)13, 'R_Collar'14, 'Head'15, 'L_Shoulder'(jian)16, 'R_Shoulder'17,
#'L_Elbow'18, 'R_Elbow'19, 'L_Wrist'20, 'R_Wrist'21, 'L_Hand'22, 'R_Hand'23]

# joints_list = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
# 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
# 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
# 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

pick = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]## 15 * 3 = 45


pick_pose = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]

joints_list = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

joints_select = ['Spine1', 'Spine2', 'Spine3', 'Neck', 'L_Collar', 'R_Collar',
       'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
       'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']


def get_pats_path(base_path, speaker, interval_id):
    interval_fn = str(interval_id) + '.h5'
    return os.path.join(base_path, speaker, interval_fn)

# joints_list = joints_list[pick]
def get_smpl_pose_path(base_path, speaker, pose_fn):
    new_pose_fn = pose_fn[:-4] + '.h5'
    return os.path.join(base_path, speaker, 'pose_smpl', new_pose_fn)

def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-b','--base_path', help="dataset root path")
parser.add_argument('-np','--num_processes', type=int, default=1)
parser.add_argument('-nf','--num_frames', type=int, default=64)
parser.add_argument('-s','--speaker', default='conan')
parser.add_argument('-pats','--pats_base_path', default='/mnt/lustressd/yuzhengming/gesture_data/pats/data/processed')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

top30_tokens = [3340, 2041, 2053, 2067, 2204, 4206, 2748 ,2157 ,2038, 8398 ,2111, 2001, 2002, 2054, 2085,
 2006, 2056, 2005, 2343, 1997, 2023, 2000, 2003, 2017, 1999,2061, 2009, 1055, 1037, 1005, 1998, 1996, 1045, 2008, 3398, 2086]
top30_words = ['stars', 'out', 'no', 'back', 'good', 'tall', 'yes', 'right', 'has', 'trump', 'people', 'was', 'he', 'what', 'now', 'on',
 'said', 'for', 'president', 'of', 'this', 'to', 'is', 'you', 'in', 'so', 'it', 's',
 'a', "'", 'and', 'the', 'i', 'that', 'yeah', 'years']

if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 可以使用use fast加速

    # # token_only = plt.subplot(111)
    # # absolute_pose = plt.subplot(323)
    # # speed_pose = plt.subplot(324)
    # # token_pose = plt.subplot(325, projection='3d')
    # -----------------------store json--------------------------------------
    # token_store = {}
    

    df = pd.read_csv(os.path.join(args.base_path, "frames_df_10_19_19.csv"))
    if args.speaker is not None:
        df = df[df['speaker'] == args.speaker]
    df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]
    # df = df[(df['dataset'] == 'dev')]


    intervals_unique = df['interval_id'].unique()
    print ("Number of unique intervals to save: %s" % (len(intervals_unique)))


    intervals = np.array_split(intervals_unique, 1)
    dfs = [df[df['interval_id'].isin(interval)] for interval in intervals]
    del df
    
    df = dfs[0]
    intervals = df['interval_id'].unique()

    # all_pose_conan = None

    #analyse mean and var of poses------------
    # for interval in tqdm(intervals):
    #     try:
    #         df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
    #         video_fn = df_interval.iloc[0]['video_fn']
    #         speaker_name = df_interval.iloc[0]['speaker']
    #         if len(df_interval) < 64:
    #             logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
    #             continue

    #         ###TODO:change poses to smpl poses
    #         poses = np.array([h5py.File(get_smpl_pose_path(args.base_path, row['speaker'], row['pose_fn']), 'r')['poses'][:] for _, row in df_interval.iterrows()])

    #         ### add audio from pats dataset
    #         pats_f = h5py.File(get_pats_path(args.pats_base_path, speaker_name, interval))
    #         audio_log_mel_400 = pats_f['audio/log_mel_400']
    #         audio_log_mel_512 = pats_f['audio/log_mel_512']

    #         ### add text from pats dataset
    #         text_w2v = pats_f['text/w2v']
    #         text_bert = pats_f['text/bert']
    #         text_token = pats_f['text/tokens']

    #         poses = torch.tensor(poses)
    #         poses = poses.float()
    #         poses = poses.to(device)
            
    #         if all_pose_conan == None:
    #             all_pose_conan = poses
    #         else:
    #             all_pose_conan = torch.cat((all_pose_conan, poses), dim=0)

    #     except Exception as e:
    #         logger.exception(e)
    #         continue

    # all_pose_conan = np.array(all_pose_conan)
    # mean_pose = torch.mean(all_pose_conan, dim=0).tolist()
    # median_pose = torch.median(all_pose_conan, dim=0)[0].tolist()
    # std_pose = torch.std(all_pose_conan, dim=0).tolist()
    # min_pose = torch.min(all_pose_conan, dim=0)[0].tolist()
    # max_pose = torch.max(all_pose_conan, dim=0)[0].tolist()

    # pose_conan = {'mean': mean_pose, 'median': mean_pose, 'std': std_pose, 'min':min_pose, 'max':max_pose}

    # str_pose_conan = json.dumps(pose_conan)
    # with open('data_conan.json', 'w') as f:
    #     f.write(str_pose_conan)

###-----------------------------draw----------------------------------
    # intervals:52723
    # plt.figure(figsize=(20.48, 20.48))

##------------save token imgs---------------------------------------------
    assert len(top30_words) == len(top30_tokens)
    for i in range(len(top30_words)):
        token = top30_tokens[i]
        word = top30_words[i]
    # ax_list = []
    # for i in range(15):
    #     plt.figure(figsize=(10.24, 10.24))
    #     ax = plt.gca()
    #     ax_list.append(ax)
    # token = 2017
    # word = 'you'
        count = 0
        while count < 10:
            try:
                idx = np.random.randint(len(intervals))

                interval = intervals[idx]
                df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
                speaker_name = df_interval.iloc[0]['speaker']
                pats_f = h5py.File(get_pats_path(args.pats_base_path, speaker_name, interval))
                text_token = pats_f['text/tokens'][()]
                # print(text_token)
                token_indexs = np.argwhere(text_token==token)
                flag = (token_indexs.shape[0] >= 5 and (token_indexs[:] - token_indexs[0])[4] == 4)

                # print(token_indexs)
                len_indexs = token_indexs.shape[0]

                if len(df_interval) < 64 or flag == False:
                    logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
                    continue
                
                poses = np.array([h5py.File(get_smpl_pose_path(args.base_path, row['speaker'], row['pose_fn']), 'r')['poses'][:] for _, row in df_interval.iterrows()])
                img_fns = np.array([get_frame_path(args.base_path, row['speaker'], row['frame_fn']) for _, row in df_interval.iterrows()])
                # print(img_fns)

                save_path = './tokens/' + word +'/' + str(interval)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for j in range(len_indexs):
                    img_fn = img_fns[token_indexs[j]]
                    img_save_path = os.path.join(save_path, '%d.jpg' % j)
                    cmd = 'cp ' + img_fn[0] + ' ' + img_save_path
                    # print(cmd)
                    subprocess.call(cmd, shell=True)
                count += 1

                #draw continue 5 frames
                # frames_pick = token_indexs[:5, 0]
                # pose_pick = poses[frames_pick]
                # pose_pick = pose_pick[:, pick_pose]
                # pose_pick = pose_pick.reshape(5, 15, 3)
                # for j in range(15):
                #     plt.sca(ax_list[j])
                #     plt.plot(range(5), pose_pick[:, j, 0], color='r')
                #     plt.plot(range(5), pose_pick[:, j, 1], color='g')
                #     plt.plot(range(5), pose_pick[:, j, 2], color='b')
                # count += 1

                #draw token2pose
                # pick_index = np.random.randint(len_indexs)
                # frame_idx = token_indexs[pick_index][0]
                # pose_pick = poses[frame_idx]
                # pose_pick = pose_pick[pick_pose]
                # pose_pick = pose_pick.reshape(15, 3)
                # #draw x, y, z
                # plt.plot(range(15), pose_pick[:, 0], 'o', color='r')
                # plt.plot(range(15), pose_pick[:, 1], '+', color='g')
                # plt.plot(range(15), pose_pick[:, 2], '*', color='b')
                # count += 1

                # #draw poses
                # frame_idx = np.random.randint(poses.shape[0])
                # pose_pick = poses[frame_idx]
                # pose_pick = pose_pick[pick_pose]
                # pose_pick = pose_pick.reshape(15, 3)
                # #draw x, y, z
                # plt.plot(range(15), pose_pick[:, 0], 'o', color='r')
                # plt.plot(range(15), pose_pick[:, 1], '+', color='g')
                # plt.plot(range(15), pose_pick[:, 2], '*', color='b')
                # count += 1

            except Exception as e:
                logger.exception(e)
                continue
##---------------------------end of save token imgs------------------------------------

            # # ## speed
            # speed = poses[1:, :] - poses[:-1, :]
            
            # ## draw speed
            # frame_idx = np.random.randint(speed.shape[0])
            # speed_pick = speed[frame_idx]
            # speed_pick = speed_pick[pick_pose]
            # speed_pick = speed_pick.reshape(15, 3)
            # #draw x, y, z
            # plt.plot(range(15), speed_pick[:, 0], 'o', color='r')
            # plt.plot(range(15), speed_pick[:, 1], '+', color='g')
            # plt.plot(range(15), speed_pick[:, 2], '*', color='b')



        # if not os.path.exists('./tokens/' + word):
        #     os.makedirs('./tokens/' + word)
        # for k in range(15):
        #     plt.sca(ax_list[k])
        #     x_major_locator=MultipleLocator(1)
        
        #     y_major_locator=MultipleLocator(0.2)
    
        #     ax=plt.gca()

        #     ax.xaxis.set_major_locator(x_major_locator)

        #     ax.yaxis.set_major_locator(y_major_locator)

        #     ax.set_xticklabels(joints_select)
        #     plt.title('token: ' + word + ' joint:' + str(k), fontsize=12)
        #     # plt.xlabel('joints_name')
        #     plt.ylabel('coordinate | r:x, g:y, b:z')
            
        #     plt.savefig('./tokens/' + word + '/' + joints_select[k] + '_anlysis.jpg')

        # x_major_locator=MultipleLocator(1)
        
        # y_major_locator=MultipleLocator(0.2)
    
        # ax=plt.gca()

        # ax.xaxis.set_major_locator(x_major_locator)

        # ax.yaxis.set_major_locator(y_major_locator)

        # ax.set_xticklabels(joints_select)
        # plt.title('token: ' + word, fontsize=12)
        # plt.xlabel('joints_name')
        # plt.ylabel('coordinate | r:x, g:y, b:z')
        # plt.savefig('token2pose(' + word + ')_anlysis.jpg')

        # plt.title('pose', fontsize=12)
        # plt.xlabel('joints_name')
        # plt.ylabel('coordinate | r:x, g:y, b:z')
        # plt.savefig('years_pose_conan_anlysis.jpg')

#----------------end of draw------------------------------------------------------

#----------------analyse tokens--------------------
    # token_store ={}
    # for interval in tqdm(intervals):
    #     try:

    #         df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
    #         if len(df_interval) < 64:
    #             logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
    #             continue           
    #         speaker_name = df_interval.iloc[0]['speaker']
    #         ## 一个iterval的数据
    #         poses = np.array([h5py.File(get_smpl_pose_path(args.base_path, row['speaker'], row['pose_fn']), 'r')['poses'][:] for _, row in df_interval.iterrows()])
    #         pats_f = h5py.File(get_pats_path(args.pats_base_path, speaker_name, interval))

    #         audio_log_mel_400 = pats_f['audio/log_mel_400']
    #         audio_log_mel_512 = pats_f['audio/log_mel_512']

    #         text_w2v = pats_f['text/w2v']
    #         text_bert = pats_f['text/bert']

    #         text_token = pats_f['text/tokens']

    #         for idx in range(0, len(df_interval)):
    #             token_name = text_token[idx]
    #             if token_store.__contains__(str(token_name)):
    #                 token_store[str(token_name)] = token_store[str(token_name)] + 1
    #             else:
    #                 token_store[str(token_name)] = 1


    #     except Exception as e:
    #         logger.exception(e)
    #         continue

    # json_token = json.dumps(token_store)
    # f = open('conan_tokens.json', 'w')
    # f.write(json_token)
    # f.close()


    # json_pose = json.dumps(pose_store)
    # f = open('analysis_pose.json', 'w')
    # f.write(json_pose)
    # f.close()

    # json_pose = json.dumps(token2pose)
    # f = open('analysis_token2pose.json', 'w')
    # f.write(json_pose)
    # f.close()


    # new_token_store = {}
    # with open ('conan_tokens.json', 'r') as f:
    #     token_store = json.load(f)
    # plt.figure(figsize=(10.24, 20.48))
    # plt.barh(list(token_store.keys()), token_store.values())            
    
    # plt.savefig('./analysis.jpg')
    # print('length of token:', token_store.__len__())
    #---------------------end of store-------------------------------
    
    #---------------------decode token to word----------------
    # new_token_store = {}
    # with open ('conan_tokens.json', 'r') as f:
    #     tokens = json.load(f)
    # for token in tokens:
    #     count = tokens[token]
    #     word = tokenizer.decode([token])
    #     new_token_store[word] = count
    
    # new_token_store = sorted(new_token_store.items(), key = lambda kv:(kv[1], kv[0]))  
    # token_sorted = sorted(tokens.items(), key = lambda kv:(kv[1], kv[0]))  

    # sort_token = json.dumps(token_sorted)
    # store = json.dumps(new_token_store)
    # with open('soted_tokens_conan.json', 'w') as f:
    #     f.write(sort_token)
    # with open('sorted_words_conan.json', 'w') as f:
    #     f.write(store)
    #-------------------end of decode-------------------------

    #-------------------save fig and count freq >= 100----------
    #5273 tokens freq >= 100, total: 18494 tokens, total counts:(all) 
    #conan: 1143 7585
    # count = 0
    # with open('sorted_words_conan.json', 'r') as f:
    #     new_token_store = json.load(f)
    # for token_pair in new_token_store:
    #     if token_pair[1] >= 100:
    #         count += 1
    # plt.figure(figsize=(20.48, 10.24))
    # new_token_store = np.array(new_token_store)
    # plt.barh(new_token_store[:, 0][-30:], new_token_store[:, 1][-30:])
    # plt.savefig('top30_freq_token_conan')
    # print(count)
    # print(new_token_store.shape[0])
    #----------------end of save---------------------------

###---------------old analyse all -------------------------------------------
    # print(dataset.__len__())#85964
    # df = pd.read_csv(os.path.join(args.base_path, 'train.csv'))
    # df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]
    # for i, row in tqdm(df.iterrows(), total=len(df)):
    #     maxCount = 0
    #     pose_fn = row['pose_fn']
    #     data = np.load(os.path.join(args.base_path, pose_fn))
    #     # logmel = data['audio_log_mel_400']
    #     ## need: tokens, poses, tokens ~ poses
        
    #     text_token = data['text_token']
    #     pose = data['pose']
    #     pose = pose[:, pick_pose]
    #     speed = pose[1:] - pose[:-1]
    #     for j in range(64):
    #         #------------------token process---------------------------
    #         token_name = text_token[j]
    #         ##huggingface.decode()
    #         if token_store.__contains__(str(token_name)):
    #             token_store[str(token_name)] = token_store[str(token_name)] + 1
    #             if token_store[str(token_name)] > maxCount:
    #                 maxCount = token_store[str(token_name)]
    #         else:
    #             token_store[str(token_name)] = 1

    #         # #-----------------pose process-------------------------------
    #         # ## pose speed
    #         # if j < 63:
    #         #     speed_points = speed[j]
    #         #     speed_points = speed_points.reshape(15, 3)
    #         #     ###draw point

    #         #     speed_pose.scatter(range(15), speed_points[:, 0], c='r')##x
    #         #     speed_pose.scatter(range(15), speed_points[:, 1], c='g')##y
    #         #     speed_pose.scatter(range(15), speed_points[:, 2], c='b')##z

    #         # ## pose absolute
    #         # points = pose[j]
    #         # points = points.reshape(15, 3) ### each points
    #         # ###daw point

    #         # absolute_pose.scatter(range(15), points[:, 0], c='r')##x
    #         # absolute_pose.scatter(range(15), points[:, 1], c='g')##y
    #         # absolute_pose.scatter(range(15), points[:, 2], c='b')##z

    #         # ### draw token2pose
    #         # token_pose.scatter([token_name]*15, range(15), points[:, 0], c='r')
    #         # token_pose.scatter([token_name]*15, range(15), points[:, 1], c='g')
    #         # token_pose.scatter([token_name]*15, range(15), points[:, 2], c='b')

    # plt.figure(1)
    # plt.barh(token_store.keys(), token_store.values())            
    
    # plt.savefig('./analysis.jpg')
    # print('length of token:', token_store.__len__())
    # print('most freq:', maxCount)
###-------------------end of old-------------------------------------------------------
