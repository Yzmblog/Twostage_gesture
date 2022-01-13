import numpy as np
from torch.cuda import device_count
from data import Dataset
import torch
from utils.average_meter import AverageMeter
from model.embedding_net import EmbeddingNet
from model.baseline import Base, Twostage
import time
import torch.nn.functional as F
from config.parse_args import parse_args


mean =  np.array([3.007988691329956, -0.0175352580845356, 0.14474990963935852, 
        -0.28565049171447754, -0.007017021998763084, 0.045262232422828674, 
        -0.46489158272743225, -0.01648695021867752, -0.07718828320503235, 
        0.24555228650569916, -0.1269848793745041, 0.01803516410291195, 
        0.3476354479789734, 0.009595228359103203, -0.06295014172792435, 
        0.6638731956481934, -0.12416858226060867, 0.033968087285757065, 
        -0.06717999279499054, -0.05199727788567543, -0.011248038150370121, 
        -0.04659409075975418, 0.25790873169898987, -0.03840641304850578, 
        -0.08543048053979874, -0.30744075775146484, 0.2294463962316513, 
        0.07968907058238983, -0.008247560821473598, -0.008622738532721996, 
        -0.296144962310791, 0.07663413137197495, 0.20712822675704956, 
        -0.2074986696243286, 0.16949273645877838, -0.5448248386383057, 
        0.06853340566158295, -0.014850771054625511, -0.060906533151865005, 
        -0.00759437819942832, 0.15292353928089142, -0.3477354645729065, 
        -0.027221398428082466, -0.07889039069414139, 0.3368067741394043, 
        0.024622777476906776, 0.05866561457514763, 0.05024004727602005, 
        0.24050311744213104, -0.12857678532600403, -1.0129417181015015, 
        0.294879287481308, 0.15455442667007446, 0.9390192627906799, 
        0.500098705291748, -1.3341306447982788, 0.29869765043258667, 
        0.2937418222427368, 1.4461997747421265, -0.14791665971279144, 
        0.026752730831503868, -0.16434234380722046, 0.010939152911305428, 
        0.08844421058893204, 0.24721364676952362, -0.00041335594141855836, 
        -0.13623714447021484, -0.1538708209991455, -0.2737543284893036, 
        -0.11275696754455566, 0.11070534586906433, 0.2602674961090088])

std = np.array([0.11005048453807831, 0.0695914477109909, 0.21332213282585144, 
0.17922855913639069, 0.04068164899945259, 0.06034838780760765, 
0.1133485734462738, 0.029268324375152588, 0.043258316814899445, 
0.08341249823570251, 0.04666789621114731, 0.020993411540985107, 
0.24318701028823853, 0.14201569557189941, 0.05194178223609924, 
0.371241956949234, 0.10448502004146576, 0.041677217930555344, 
0.06540325284004211, 0.03929987549781799, 0.023001153022050858, 
0.08263451606035233, 0.11042477935552597, 0.0826077088713646, 
0.10972023010253906, 0.049007806926965714, 0.09652846306562424, 
0.06505760550498962, 0.053020402789115906, 0.017971057444810867, 
0.06829532980918884, 0.10542648285627365, 0.07906108349561691, 
0.07396340370178223, 0.07366187125444412, 0.11737411469221115, 
0.1999451220035553, 0.10558865964412689, 0.08671864122152328, 
0.064442940056324, 0.13395139575004578, 0.13020926713943481, 
0.054195601493120193, 0.1408957690000534, 0.17446929216384888, 
0.17152321338653564, 0.09445513784885406, 0.09816849231719971, 
0.19257396459579468, 0.11027055978775024, 0.13003741204738617, 
0.15299145877361298, 0.10161574929952621, 0.17045913636684418, 
0.2611887454986572, 0.47025159001350403, 0.20657111704349518, 
0.29691192507743835, 0.4792603850364685, 0.2602325677871704, 
0.19745808839797974, 0.12035299092531204, 0.11156259477138519, 
0.13563837110996246, 0.159286230802536, 0.14487288892269135, 
0.10297834873199463, 0.07652372866868973, 0.10554931312799454, 
0.04732266813516617, 0.02601797878742218, 0.08198181539773941])

pick_pose = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]

mean = mean[pick_pose]
std = std[pick_pose]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_pck(pred, gt, alpha=0.2):
    '''
    :param pred: predicted keypoints on NxMxK where N is number of samples, M is of shape 3, corresponding to X,Y and K is the number of keypoints to be evaluated on
    :param gt:  similarly
    :param alpha: parameters controlling the scale of the region around the image multiplied by the max(H,W) of the person in the image. We follow https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1
    :return: mean prediction score
    '''
    pck_radius = compute_pck_radius(gt, alpha)
    keypoint_overlap = (np.linalg.norm(np.transpose(gt-pred, [0, 2, 1]), axis=2) <= (pck_radius))
    # print('pred shape', pred.shape, 'gt_shape', gt.shape, 'pck_radius_shape', pck_radius.shape, 'key_overlap_shape',keypoint_overlap.shape)
    return np.mean(keypoint_overlap, axis=1)


def compute_pck_radius(gt, alpha):
    x = np.abs(np.max(gt[:, 0:1], axis=2) - np.min(gt[:, 0:1], axis=2))
    y = np.abs(np.max(gt[:, 1:2], axis=2) - np.min(gt[:, 1:2], axis=2))
    z = np.abs(np.max(gt[:, 2:3], axis=2) - np.min(gt[:, 2:3], axis=2))

    max_axis = np.concatenate([x, y, z], axis=1).max(axis=1)
    max_axis_per_keypoint = np.tile(np.expand_dims(max_axis, -1), [1, 15])
    return max_axis_per_keypoint * alpha


def main(args):
    dataset_dev = Dataset(args=args, type_data='dev')
    dataset_test = Dataset(args=args, type_data='test')
    # print(dataset_dev.__len__())# 9713
    dataloader_dev = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    checkpoint = torch.load(args.load_from_path, map_location=device)
    generator = Twostage(args).to(device)
    generator.load_state_dict(checkpoint['gen_dict'])

    # loss_fn = torch.nn.L1Loss()
    # ckpt = torch.load(args.eval_net_path, map_location=device)
    # n_frames = args.n_poses
    # mode = 'pose'
    # pose_dim = ckpt['pose_dim']
    # embedding_net = EmbeddingNet(args, pose_dim, n_frames, None, args.wordembed_dim,
    #                         None, mode).to(device)
    # embedding_net.load_state_dict(ckpt['gen_dict'])
    # embedding_net.train(False)

    # to evaluation mode
    generator.train(False)

    # if embed_space_evaluator:
    #     embed_space_evaluator.reset()
    l1_loss_dev = AverageMeter('l1_loss_dev')
    # joint_mae = AverageMeter('mae_on_joint')
    speed_dev = AverageMeter('speed_dev')
    accel_dev = AverageMeter('accel_dev')
    pck_dev = AverageMeter('pck_dev')

    l1_loss_test = AverageMeter('l1_loss_test')
    # joint_mae = AverageMeter('mae_on_joint')
    speed_test = AverageMeter('speed_test')
    accel_test = AverageMeter('accel_test')
    pck_test = AverageMeter('pck_test')

    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader_dev, 0):

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
            # speaker_model = utils.train_utils.get_speaker_model(generator)
            # if speaker_model:
            #     vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)
            # else:
            #     vid_indices = None

            # synthesize //now try no pre pose
            # pre_seq = target.new_zeros((target.shape[0], target.shape[1],
            #                                     target.shape[2] + 1))
            # pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            # pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
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
            # if args.model == 'multimodal_context':
            #     pred_pose, *_ = generator(pre_seq, in_text, in_audio, vid_indices)
            #     loss = F.l1_loss(pred_pose, target)
            # elif args.model == 'gesture_autoencoder':
            #     loss, _ = eval_embed(in_text, in_audio, None, target, generator)
            # elif args.model == 'baseline':
            #     pred_pose = generator(in_audio=in_audio, in_text=in_text)
            # #     loss = F.l1_loss(pred_pose, target)
            # elif args.model == 'Twostage':
            pred_pose, _, _ = generator(in_audio=in_audio, in_text=in_text)
            pred_pose = (pred_pose * (std + np.finfo(float).eps)) + mean
            target = (target * (std + np.finfo(float).eps)) + mean

            loss = F.l1_loss(pred_pose, target)
            # pred_pose = generator(in_audio=in_audio, in_text=in_text)
            # loss = F.l1_loss(pred_pose, target)

            l1_loss_dev.update(loss.item(), batch_size)


            ###TODO:1.change init model
            ###     2.try some gan and other nets
            # if args.model != 'gesture_autoencoder':
            #     if embed_space_evaluator:
            #         embed_space_evaluator.push_samples(in_text, in_audio, pred_pose, target)

            np_pred = pred_pose.cpu().numpy()
            np_gt = target.cpu().numpy()

            pck_score = compute_pck(np_pred.reshape((-1, 3, 15)), np_gt.reshape((-1, 3, 15)))
            pck_score = np.mean(pck_score)
            pck_dev.update(pck_score, batch_size * 64)

            # diff = np_pred - np_gt
            # mae_val = np.mean(np.absolute(diff))
            # joint_mae.update(mae_val, batch_size)

            # spped
            target_speed = np.diff(np_gt, n=1, axis=1)
            out_speed = np.diff(np_pred, n=1, axis=1)
            speed_dev.update(np.mean(np.abs(target_speed - out_speed)), batch_size)

            # accel
            target_acc = np.diff(np_gt, n=2, axis=1)### accel(n = 2: er jie dao)
            out_acc = np.diff(np_pred, n=2, axis=1)
            accel_dev.update(np.mean(np.abs(target_acc - out_acc)), batch_size)
    dev_time = time.time() - start
    print('evaluation in dev:', ' l1:', l1_loss_dev.avg, ' pck:', pck_dev.avg, ' speed:', speed_dev.avg, ' accel:', accel_dev.avg)

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader_test, 0):

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
            # speaker_model = utils.train_utils.get_speaker_model(generator)
            # if speaker_model:
            #     vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)
            # else:
            #     vid_indices = None

            # synthesize //now try no pre pose
            # pre_seq = target.new_zeros((target.shape[0], target.shape[1],
            #                                     target.shape[2] + 1))
            # pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            # pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
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
            # if args.model == 'multimodal_context':
            #     pred_pose, *_ = generator(pre_seq, in_text, in_audio, vid_indices)
            #     loss = F.l1_loss(pred_pose, target)
            # elif args.model == 'gesture_autoencoder':
            #     loss, _ = eval_embed(in_text, in_audio, None, target, generator)
            # elif args.model == 'baseline':
            #     pred_pose = generator(in_audio=in_audio, in_text=in_text)
            # #     loss = F.l1_loss(pred_pose, target)
            # elif args.model == 'Twostage':
            pred_pose, _, _ = generator(in_audio=in_audio, in_text=in_text)
            pred_pose = (pred_pose * (std + np.finfo(float).eps)) + mean
            target = (target * (std + np.finfo(float).eps)) + mean
            loss = F.l1_loss(pred_pose, target)
            # pred_pose = generator(in_audio=in_audio, in_text=in_text)
            # loss = F.l1_loss(pred_pose, target)

            l1_loss_test.update(loss.item(), batch_size)


            ###TODO:1.change init model
            ###     2.try some gan and other nets
            # if args.model != 'gesture_autoencoder':
            #     if embed_space_evaluator:
            #         embed_space_evaluator.push_samples(in_text, in_audio, pred_pose, target)

            np_pred = pred_pose.cpu().numpy()
            np_gt = target.cpu().numpy()

            pck_score = compute_pck(np_pred.reshape((-1, 3, 15)), np_gt.reshape((-1, 3, 15)))
            pck_score = np.mean(pck_score)
            pck_test.update(pck_score, batch_size * 64)

            # diff = np_pred - np_gt
            # mae_val = np.mean(np.absolute(diff))
            # joint_mae.update(mae_val, batch_size)

            # spped
            target_speed = np.diff(np_gt, n=1, axis=1)
            out_speed = np.diff(np_pred, n=1, axis=1)
            speed_test.update(np.mean(np.abs(target_speed - out_speed)), batch_size)

            # accel
            target_acc = np.diff(np_gt, n=2, axis=1)### accel(n = 2: er jie dao)
            out_acc = np.diff(np_pred, n=2, axis=1)
            accel_test.update(np.mean(np.abs(target_acc - out_acc)), batch_size)
    print('evaluation in test:', ' l1:', l1_loss_test.avg, ' pck:', pck_test.avg, ' speed:', speed_test.avg, ' accel:', accel_test.avg)

if __name__ == '__main__':
    args = parse_args()
    main(args)
