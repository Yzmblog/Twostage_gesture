# from model.Audio2Gesture import A2GNet
import argparse
from pickle import NONE
from re import L, S
import sys

from torch.nn.modules import loss
[sys.path.append(i) for i in ['.', '..']]
from matplotlib.pyplot import axis, text
from model.baseline import Base, Twostage
# from data_loader import Data
# from scripts.data_loader import audio
from utils.average_meter import AverageMeter
import time
from loss.loss import calculate_loss, base_loss
import utils.train_utils
import os
import logging
import torch
import pprint
from config.parse_args import parse_args
import datetime
from pathlib import Path
from torch import optim
import torch.nn.functional as F
import numpy as np
from data import Dataset
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.embedding_net import EmbeddingNet
from model.seq2seq_net import Seq2SeqNet
from model import speech2gesture, vocab
from train_eval.train_speech2gesture import train_iter_speech2gesture
from train_eval.train_gan import train_iter_gan
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from train_eval.train_seq2seq import train_iter_seq2seq
from train_eval.train_speech2gesture import train_iter_speech2gesture
from utils.vocab_utils import build_vocab
import random

from torch.utils.tensorboard import SummaryWriter

comb = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 
print(device)

def train_iter_base(args, in_text, in_audio, target, generator, gen_optimizer, loss_fn):
    output = generator(in_audio=in_audio, in_text=in_text)
    # loss = []
    gen_optimizer.zero_grad()

    # output_speed = output[1:] - output[:-1]
    # target_speed = target[1:] - target[:-1]

    l1_loss = loss_fn(output, target)

    # total_loss = l1_loss
    l1_loss.backward()
    gen_optimizer.step()

    return {'l1': l1_loss}

def train_iter_Twostage(args, in_text, in_audio, target, generator, gen_optimizer, loss_fn, embedding_net):
    pred_pose, pred_speed, propasal_pose = generator(in_audio=in_audio, in_text=in_text)
    gen_optimizer.zero_grad()
    embedding_net.train(False)


    l1_pose = loss_fn(pred_pose, target)
    #l1_speed = loss_fn(pred_speed, target)
    # propasal_diff_loss = 0

    # for idxs in comb:
    #     loss = loss_fn(propasal_pose[:, idxs[0], :, ], propasal_pose[:, idxs[1], :, ])
    #     propasal_diff_loss += loss
    
    feat_loss = 0
    with torch.no_grad():
        _, _, _, real_feat, _, _, _ = embedding_net(None, None, None, target,
                                                                'pose', variational_encoding=False)
    
    for i in range(propasal_pose.shape[1]):

        _, _, _, generated_feat, _, _, _ = embedding_net(None, None, None, propasal_pose[:, i, :, ],
                                                                  'pose', variational_encoding=False)
        loss = loss_fn(generated_feat, real_feat)

        feat_loss += loss

    total_loss = l1_pose + 0.0167 * feat_loss ##+ l1_speed## -  0.066 * propasal_diff_loss
    total_loss.backward()
    gen_optimizer.step()

    return {'l1':l1_pose, 'total': total_loss, 'feat_loss': feat_loss}##, 'speed': l1_speed}


def init_model(args, lang_model, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = embedding_net = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  lang_model=lang_model,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
    elif args.model == 'joint_embedding':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='random').to(_device)
    elif args.model == 'gesture_autoencoder':
        generator = EmbeddingNet(args, pose_dim, n_frames, None, None,
                                 None, mode='pose').to(_device)
    elif args.model == 'seq2seq':
        generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                               lang_model.word_embedding_weights).to(_device)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'speech2gesture':
        generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses).to(_device)
        discriminator = speech2gesture.Discriminator(pose_dim).to(_device)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'baseline':
        generator = Base(args).to(_device)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'Twostage':
        generator = Twostage(args).to(_device)
        loss_fn = torch.nn.L1Loss()
        ckpt = torch.load(args.eval_net_path, map_location=device)
        n_frames = args.n_poses
        mode = 'pose'
        pose_dim = ckpt['pose_dim']
        embedding_net = EmbeddingNet(args, pose_dim, n_frames, None, args.wordembed_dim,
                                None, mode).to(device)
        embedding_net.load_state_dict(ckpt['gen_dict'])
        embedding_net.train(False)
    return generator, discriminator, loss_fn, embedding_net


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim=45, speaker_model=None):
    start = time.time()
    # loss_meters = [AverageMeter('l1'), AverageMeter('speed'), AverageMeter('accl')]
    loss_meters = [AverageMeter('l1'), AverageMeter('speed'), AverageMeter('total'), AverageMeter('feat_loss'), AverageMeter('proposal'),
    AverageMeter('accl'), AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + \
        str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(
        Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 20)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 2

    # z type
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None


    # init model
    if args.load_from_path == None:
        generator, discriminator, loss_fn, embedding_net = init_model(args, lang_model, speaker_model, pose_dim, device)
    else:
        checkpoint = torch.load(args.load_from_path)
        if args.model == 'baseline':
            generator = Base(args).to(device)
            loss_fn = torch.nn.L1Loss()
            discriminator = None
        elif args.model == 'gesture_autoencoder':
            generator = EmbeddingNet(args, pose_dim, args.n_poses, lang_model.n_words, args.wordembed_dim,
                                    lang_model.word_embedding_weights, mode='pose').to(device)
            discriminator = loss_fn = None
        elif args.model == 'Twostage':
            generator = Twostage(args).to(device)
            loss_fn = torch.nn.L1Loss()
            discriminator = None
            ckpt = torch.load(args.eval_net_path, map_location=device)
            n_frames = args.n_poses
            mode = 'pose'
            pose_dim = ckpt['pose_dim']
            embedding_net = EmbeddingNet(args, pose_dim, n_frames, None, args.wordembed_dim,
                                    None, mode).to(device)
            embedding_net.load_state_dict(ckpt['gen_dict'])
            embedding_net.train(False)
            generator.load_state_dict(checkpoint['gen_dict'])
    # use multi GPUs
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)


    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    gen_optimizer = optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.num_epochs):
        # evaluate the test set
        val_metrics = evaluate_testset(args, test_data_loader, generator, embed_space_evaluator=embed_space_evaluator)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation',
                                 val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        if 'frechet' in val_metrics.keys():
            val_loss = val_metrics['frechet']
        else:
            val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(
                best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(
                    args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(
                    args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict,
            }, save_name)

        # # save sample results
        # if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
        #     evaluate_sample_and_save_video(
        #         epoch, args.name, test_data_loader, generator,
        #         args=args, lang_model=lang_model)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_audio = data['audio']
            in_text = data['text']
            target = data['pose']

            batch_size = target.size(0)

            in_audio = in_audio.float()
            if args.model != 'multimodal_context':
                in_text = in_text.float()
            target = target.float()


            in_audio = in_audio.to(device)

            if args.model != 'multimodal_context':
                in_text = in_text.to(device)
            target = target.to(device)

            # speaker input
            vid_indices = []
            # if speaker_model and isinstance(speaker_model, vocab.Vocab):
            #     vids = aux_info['vid']##change
            #     vid_indices = [speaker_model.word2index[vid] for vid in vids]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)


            # train
            loss = []
            if args.model == 'multimodal_context':
                loss = train_iter_gan(args, epoch, in_text, in_audio, target, vid_indices,
                                      generator, discriminator,
                                      gen_optimizer, dis_optimizer)
            elif args.model == 'joint_embedding':
                loss = train_iter_embed(args, epoch, in_text, in_audio, target,
                                        generator, gen_optimizer, mode='random')
            elif args.model == 'gesture_autoencoder':
                loss = train_iter_embed(args, epoch, in_text, in_audio, target,
                                        generator, gen_optimizer)
            # elif args.model == 'seq2seq':
            #     loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target, generator, gen_optimizer)
            elif args.model == 'speech2gesture':
                loss = train_iter_speech2gesture(args, in_audio, target, generator, discriminator,
                                                 gen_optimizer, dis_optimizer, loss_fn)
            elif args.model == 'baseline':
                loss = train_iter_base(args, in_text, in_audio, target, generator, gen_optimizer, loss_fn)
            elif args.model == 'Twostage':

                loss = train_iter_Twostage(args, in_text, in_audio, target, generator, gen_optimizer, loss_fn, embedding_net)
                

            #----------old code for base------------------------------
            # output = generator(in_audio=in_audio, in_text=in_text)
            # # loss = []
            # gen_optimizer.zero_grad()

            # loss = base_loss(output, target)

            # l1_loss = loss['l1']
            # total_loss = l1_loss
            # total_loss.backward()
            # gen_optimizer.step()
            #-----------end of old code---------------------------------



            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(
                            loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(
            key, best_values[key][0], best_values[key][1]))



def evaluate_testset(args, test_data_loader, generator, loss_fn=None, embed_space_evaluator=None):
    # to evaluation mode
    generator.train(False)

    # if embed_space_evaluator:
    #     embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    speed = AverageMeter('speed')
    accel = AverageMeter('accel')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):

            target = data['pose']
            in_audio = data['audio']
            in_text = data['text']
            in_audio = in_audio.float()
            if args.model != 'multimodal_context':
                in_text = in_text.float()
            target = target.float()

            batch_size = target.size(0)
            in_audio = in_audio.to(device)  # audio: raw audio
            if args.model != 'multimodal_context':
                in_text = in_text.to(device)
            target = target.to(device)
            
            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            # synthesize //now try no pre pose
            pre_seq = target.new_zeros((target.shape[0], target.shape[1],
                                                target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            # pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            # if args.model == 'joint_embedding':
            #     loss, out_dir_vec = eval_embed(in_text_padded, in_audio, pre_seq_partial,
            #                                    target, generator, mode='speech')
            # elif args.model == 'gesture_autoencoder':
            #     loss, _ = eval_embed(in_text_padded, in_audio, pre_seq_partial, target, generator)
            # elif args.model == 'seq2seq':
            #     out_dir_vec = generator(in_text, text_lengths, target, None)
            #     loss = loss_fn(out_dir_vec, target)
            # elif args.model == 'speech2gesture':
            #     out_dir_vec = generator(in_spec, pre_seq_partial)
            #     loss = loss_fn(out_dir_vec, target)
            if args.model == 'multimodal_context':
                pred_pose, *_ = generator(pre_seq, in_text, in_audio, vid_indices)
                loss = F.l1_loss(pred_pose, target)
            elif args.model == 'gesture_autoencoder':
                loss, _ = eval_embed(in_text, in_audio, None, target, generator)
            elif args.model == 'baseline':
                pred_pose = generator(in_audio=in_audio, in_text=in_text)
                loss = F.l1_loss(pred_pose, target)
            elif args.model == 'Twostage':
                pred_pose, _, _ = generator(in_audio=in_audio, in_text=in_text)
                loss = F.l1_loss(pred_pose, target)
            # pred_pose = generator(in_audio=in_audio, in_text=in_text)
            # loss = F.l1_loss(pred_pose, target)

            losses.update(loss.item(), batch_size)


            ###TODO:1.change init model
            ###     2.try some gan and other nets
            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text, in_audio, pred_pose, target)

                np_pred = pred_pose.cpu().numpy()
                np_gt = target.cpu().numpy()

                diff = np_pred - np_gt
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # spped
                target_speed = np.diff(np_gt, n=1, axis=1)
                out_speed = np.diff(np_pred, n=1, axis=1)
                speed.update(np.mean(np.abs(target_speed - out_speed)), batch_size)

                # accel
                target_acc = np.diff(np_gt, n=2, axis=1)### accel(n = 2: er jie dao)
                out_acc = np.diff(np_pred, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg, 'speed': speed.avg, 'accl': accel.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        logging.info(
            '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
    else:
        logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s, speed diff: {:.5f}, accel diff: {:.5f}'.format(
            losses.avg, joint_mae.avg, elapsed_time, speed.avg, accel.avg))

    return ret_dict



def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(
        args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(
        torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))


###---------------------old pats dataset loader------------------------------------------###
#    # dataset                                                                             #
#    # common_kwargs = dict(path2data='/mnt/lustressd/yuzhengming/gesture_data/pats/data', #
#    #                      speaker=['all'],                                               #
#    #                      modalities=['pose/data', 'audio/log_mel_512'],                 #
#    #                      fs_new=[15, 15, 15],                                           #
#    #                      batch_size=128,                                                #
#    #                      window_hop=5,                                                  #
#    #                      num_workers=0)                                                 #
#    # data = Data(**common_kwargs)                                                        #
#    # train_loader = data.train                                                           #
#    # test_loader = data.dev                                                              #
#                                                                                          #
#    # train                                                                               #
#    #train_epochs(args, train_loader, test_loader)                                        #
###----------------------end of old pats dataset loader------------------------------------#

###load training dataset
    dataset_train = Dataset(args=args, type_data='train')
    # print(dataset.__len__())#85964
    dataset_dev = Dataset(args=args, type_data='dev')
    # print(dataset_dev.__len__())# 9713
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    dataloader_dev = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True  
    )

    lang_model = None
    if args.model in ['baseline', 'multimodal_context']:
        vocab_cache_path = os.path.join(os.path.split(args.base_path)[0], 'vocab_cache.pkl')
        lang_model = build_vocab('words', [dataset_train, dataset_dev, dataset_dev], vocab_cache_path, args.wordembed_path,
                            args.wordembed_dim)
    train_epochs(args, dataloader_train, dataloader_dev, lang_model)



if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
