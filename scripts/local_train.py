from model.Audio2Gesture import A2GNet
from data_loader import Data
from utils.average_meter import AverageMeter
import time
from loss.loss import calculate_loss
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
import sys
[sys.path.append(i) for i in ['.', '..']] ### Add search path of 'import'
from torch.utils.tensorboard import SummaryWriter
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from utils.vocab_utils import build_vocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_testset(args, test_data_loader, generator, loss_fn=None, embed_space_evaluator=None):
    # to evaluation mode
    generator.train(False)

    # if embed_space_evaluator:
    #     embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):

            target = data['pose/data']
            in_audio = data['audio/log_mel_512']
            in_audio = in_audio.float()
            batch_size = target.size(0)
            target = target.float()

            # in_text = in_text.to(device)
            # in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)  # audio: raw audio
            # in_spec = in_spec.to(device) ##spec: audio feat
            target = target.to(device)

            # speaker input
            # speaker_model = utils.train_utils.get_speaker_model(generator)
            # if speaker_model:
            #     vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)
            # else:
            #     vid_indices = None

            sample_vectors = [torch.randn((batch_size, 16, 64)).to(
                device) for _ in range(2)]  # change to normal of mean and varience
            pred_pose = generator(in_audio, sample_vectors)[2]
            loss = F.l1_loss(pred_pose, target)

            losses.update(loss.item(), batch_size)

            if args.model != 'gesture_autoencoder':
                # if embed_space_evaluator:
                #     embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                # out_dir_vec = out_dir_vec.cpu().numpy()
                # out_dir_vec += np.array(args.mean_dir_vec).squeeze()
                # out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                # target_vec = target_vec.cpu().numpy()
                # target_vec += np.array(args.mean_dir_vec).squeeze()
                # target_poses = convert_dir_vec_to_pose(target_vec)

                # if out_joint_poses.shape[1] == args.n_poses:
                #     diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                # else:
                #     diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                out_joint_poses = pred_pose.reshape(
                    (pred_pose.shape[0], pred_pose.shape[1], -1, 2))
                target_poses = target.reshape(
                    (target.shape[0], target.shape[1], -1, 2))
                out_joint_poses = out_joint_poses.cpu().numpy()
                target_poses = target_poses.cpu().numpy()
                diff = out_joint_poses - target_poses
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    # if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
    #     frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    #     logging.info(
    #         '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
    #             losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, elapsed_time))
    #     ret_dict['frechet'] = frechet_dist
    #     ret_dict['feat_dist'] = feat_dist
    # else:
    logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
        losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict


def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, args,
                                   n_save=None, save_path=None):
    generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    out_raw = []

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break

            target = data['pose/data']
            in_audio = data['audio/log_mel_512']
            in_audio = in_audio.float()
            target = target.float()

            # prepare
            select_index = 0
            # if args.model == 'seq2seq':
            #     in_text = in_text[select_index, :].unsqueeze(0).to(device)
            #     text_lengths = text_lengths[select_index].unsqueeze(0).to(device)
            # in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            # in_spec = in_spec[select_index, :, :].unsqueeze(0).to(device)
            target = target[select_index, :, :].unsqueeze(0).to(device)

            # input_words = []
            # for i in range(in_text_padded.shape[1]):
            #     word_idx = int(in_text_padded.data[select_index, i])
            #     if word_idx > 0:
            #         input_words.append(lang_model.index2word[word_idx])
            # sentence = ' '.join(input_words)

            # speaker input
            # speaker_model = utils.train_utils.get_speaker_model(generator)
            # if speaker_model:
            #     vid = aux_info['vid'][select_index]
            #     # vid_indices = [speaker_model.word2index[vid]]
            #     vid_indices = [random.choice(list(speaker_model.word2index.values()))]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)
            # else:
            vid_indices = None

            aux_info = data['meta']
            # aux info
            aux_str = '({}, time: {}-{})'.format(
                aux_info['interval_id'][select_index],
                str(datetime.timedelta(seconds=float(
                    aux_info['start'][select_index]))),
                str(datetime.timedelta(seconds=float(aux_info['end'][select_index]))))

            # synthesize
            # pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1],
            #                                     target_dir_vec.shape[2] + 1))
            # pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            # pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            # pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            # if args.model == 'multimodal_context':
            #     out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices)
            # elif args.model == 'joint_embedding':
            #     _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
            # elif args.model == 'gesture_autoencoder':
            #     _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, target_dir_vec,
            #                                               variational_encoding=False)
            # elif args.model == 'seq2seq':
            #     out_dir_vec = generator(in_text, text_lengths, target_dir_vec, None)
            #     # out_poses = torch.cat((pre_poses, out_poses), dim=1)
            # elif args.model == 'speech2gesture':
            batch_size = target.shape[0]
            sample_vectors = [torch.randn((batch_size, 16, 64)).to(
                device) for _ in range(2)]  # change to normal of mean and varience
            out_poses = generator(in_audio, sample_vectors)[2]

            # to video
            audio_npy = np.squeeze(in_audio.cpu().numpy())
            target = np.squeeze(target.cpu().numpy())
            out_poses = np.squeeze(out_poses.cpu().numpy())
            # print(out_poses.shape)
            if save_path is None:
                save_path = args.model_save_path

            # mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)

            out_poses = out_poses.reshape(
                (out_poses.shape[0], -1, 2))
            target = target.reshape(
                (target.shape[0], -1, 2))
            utils.train_utils.create_video_and_save(
                save_path, epoch, prefix, iter_idx,
                target, out_poses,
                audio=audio_npy, aux_str=aux_str)

            # target_dir_vec = target_dir_vec.reshape((target_dir_vec.shape[0], 9, 3))
            # out_dir_vec = out_dir_vec.reshape((out_dir_vec.shape[0], 9, 3))
            out_raw.append({
                # 'sentence': sentence,
                'audio': audio_npy,
                'human_dir_vec': target,
                'out_dir_vec': out_poses,
                'aux_info': aux_str
            })

    generator.train(True)  # back to training mode
    logging.info('saved sample videos, took {:.1f}s'.format(
        time.time() - start))

    return out_raw


def train_epochs(args, train_data_loader, test_data_loader):
    start = time.time()
    loss_meters = [AverageMeter('rot_motion'), AverageMeter('speed_motion'), AverageMeter('pos_motion'), AverageMeter('motion1'), AverageMeter('motion2'), AverageMeter('relax'),
                 AverageMeter('total')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + \
        str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(
        Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 20)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 5

    # init model
    if args.model == 'A2G':
        generator = A2GNet(args).to(device)
    else:
        speaker_model = None
        vocab_cache_path = './vocab_cache.pkl'
        lang_model = build_vocab('words', [train_data_loader, test_data_loader, test_data_loader], vocab_cache_path,
                                 args.wordembed_path,
                                 args.wordembed_dim)
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=args.pose_dim).to(device)
        discriminator = ConvDiscriminator(args.pose_dim).to(device)

    # use multi GPUs
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    # embed_space_evaluator = None
    # if args.eval_net_path and len(args.eval_net_path) > 0:
    #     embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    # gen_optimizer = optim.Adam(
    #     generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    # dis_optimizer = None
    # if discriminator is not None:
    #     dis_optimizer = torch.optim.Adam(discriminator.parameters(),
    #                                      lr=args.learning_rate * args.discriminator_lr_weight,
    #                                      betas=(0.5, 0.999))
    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
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
        val_metrics = evaluate_testset(args, test_data_loader, generator)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation',
                                 val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        # if 'frechet' in val_metrics.keys():
        #     val_loss = val_metrics['frechet']
        # else:
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
            # dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                # if discriminator is not None:
                #     dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                # if discriminator is not None:
                #     dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(
                    args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(
                    args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'gen_dict': gen_state_dict,
                # 'dis_dict': dis_state_dict,
            }, save_name)

        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name, test_data_loader, generator,
                args=args)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_audio = data['audio/log_mel_512']
            target = data['pose/data']
            batch_size = target.size(0)

            in_audio = in_audio.float()
            # target = target.float()
            # in_text = in_text.to(device)
            # in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            # in_spec = in_spec.to(device)
            target = target.to(device)

            # speaker input
            # vid_indices = []
            # if speaker_model and isinstance(speaker_model, vocab.Vocab):
            #     vids = aux_info['vid']
            #     vid_indices = [speaker_model.word2index[vid] for vid in vids]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)

            # train
            sample_vectors = [torch.randn(
                (batch_size, 16, 64)).to(device) for _ in range(2)]

            output = generator(in_audio, sample_vectors, target)
            # loss = []
            gen_optimizer.zero_grad()

            loss = calculate_loss(output, target)


            total_loss = loss['total']
            total_loss.backward()
            gen_optimizer.step()
            # if args.model == 'multimodal_context':
            #     loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
            #                           generator, discriminator,
            #                           gen_optimizer, dis_optimizer)
            # elif args.model == 'joint_embedding':
            #     loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
            #                             generator, gen_optimizer, mode='random')
            # elif args.model == 'gesture_autoencoder':
            #     loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
            #                             generator, gen_optimizer)
            # elif args.model == 'seq2seq':
            #     loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)
            # elif args.model == 'speech2gesture':
            #     loss = train_iter_speech2gesture(args, in_spec, target_vec, generator, discriminator,
            #                                      gen_optimizer, dis_optimizer, loss_fn)

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

    # dataset
    common_kwargs = dict(path2data='/media/SENSETIME_yuzhengming/DATA/PATS/all/data',
                         speaker=['chemistry'],
                         modalities=['pose/data', 'audio/log_mel_512', 'text/w2v'],
                         fs_new=[15, 15, 15],
                         batch_size=2,
                         window_hop=5,
                         num_workers=0,
                         load_data=False)
    data = Data(**common_kwargs)
    train_loader = data.train
    test_loader = data.dev

    # train
    train_epochs(args, train_loader, test_loader)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
