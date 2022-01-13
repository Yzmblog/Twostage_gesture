from operator import pos
import torch
import torch.nn.functional as F
from utils.data_utils import convert_pose_seq_to_dir_vec
import numpy as np

### just calculate l1 base loss first
def base_loss(pred, target):
    l1_loss = F.l1_loss(pred, target)

    ###speed
    # target_speed = target[:, 1:, :] - target[:, :-1, :]
    # pred_speed = pred[:, 1:, :] - pred[:, :-1, :]
    # speed_loss = F.l1_loss(pred_speed, target_speed)

    # # accel
    # target_acc = target_speed[:, 1:, :] - target_speed[:, :-1, :]
    # pred_acc = pred_speed[:, 1:, :] - pred_speed[:, :-1, :]
    # accl_loss = F.l1_loss(pred_acc, target_acc)

    return {'l1': l1_loss}

def calculate_loss(pred, target):
    pose_mm, pose_am, pose_ar1, pose_ar2, share_code_motion, share_code_audio, recon_code1, sample_code1, motion_sample = pred
    # print(pose_mm.shape, pose_am.shape, pose_ar1.shape, pose_ar2.shape, share_code_motion.shape, share_code_audio.shape)


    ###loss
    # loss_cyc = F.l1_loss(recon_code1, sample_code1)
    # beta = 0.1
    # huber_loss = F.smooth_l1_loss(pose_am / beta, target / beta) * beta

    ##改成cos?
    # loss_share_align = F.l1_loss(share_code_audio, share_code_motion)
    loss_audio = F.l1_loss(share_code_audio, share_code_motion)
    loss_random = F.l1_loss(sample_code1, motion_sample)


    # loss_relax = max(F.l1_loss(pose_ar1, target) - 0.02, 0)
    # loss_div = -F.l1_loss(pose_ar1, pose_ar2)
    # loss_div = 0

    # loss_pos_motion = F.l1_loss(pose_mm, target)
    # loss_pos_audio = F.l1_loss(pose_am, target)

    # dir_mm = convert_pose_seq_to_dir_vec(pose_mm)
    # dir_am = convert_pose_seq_to_dir_vec(pose_am)

    # target_dir = convert_pose_seq_to_dir_vec(target)

    # print(dir_mm.shape)

    ### calculate rotation xita
    # dir_mm = pose_mm.reshape((pose_mm.shape[0], -1, 2))
    # # dir_am = pose_am.reshape((pose_am.shape[0], -1, 2))
    # target_dir = target.reshape((target.shape[0], -1, 2))
    #
    # loss_rot_motion = torch.mean(1 - torch.cosine_similarity(dir_mm, target_dir, dim=-1))
    # loss_rot_audio = torch.mean(1 - torch.cosine_similarity(dir_am, target_dir, dim=-1))

    ### calculate speed
    # speed_mm = pose_mm[:, 1:, :] - pose_mm[:, :-1, :]
    # # speed_am = pose_am[:, 1:, :] - pose_am[:, :-1, :]
    # target_speed = target[:, 1:, :] - target[:, :-1, :]
    #
    # loss_speed_motion = F.l1_loss(speed_mm, target_speed)
    # # loss_speed_audio = F.l1_loss(speed_am, target_speed)
    #
    # loss_motion1 = loss_rot_motion + loss_pos_motion + 5 * loss_speed_motion
    # loss_motion2 = loss_rot_audio + loss_pos_audio + 5 * loss_speed_audio

    # total_loss = loss_motion1 + loss_motion2 + loss_relax + loss_cyc + loss_share_align
    total_loss = loss_audio + loss_random



    return {'rot_motion':0, 'rot_audio':0, 'speed_motion':0,
    'speed_audio': 0, 'pos_motion': 0, 'pos_audio': 0,
    'motion1': 0, 'motion2': 0, 'relax': 0, 'cyc': 0, 'align': 0, 'total': total_loss, 'audio': loss_audio, 'random': loss_random}
