from pickle import NONE
from numpy import not_equal
from torch.serialization import default_restore_location
import configargparse
import logging

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    # parser.add("--train_data_path", action="append")
    # parser.add("--val_data_path", action="append")
    # parser.add("--test_data_path", action="append")
    # parser.add("--model_save_path", required=True)
    # parser.add("--pose_representation", type=str, default='3d_vec')
    # parser.add("--mean_dir_vec", action="append", type=float, nargs='*')
    # parser.add("--mean_pose", action="append", type=float, nargs='*')
    # parser.add("--random_seed", type=int, default=-1)
    # parser.add("--save_result_video", type=str2bool, default=True)
    #
    # # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=100)
    parser.add("--freeze_wordembed", type=str2bool, default=False)
    #
    # # model
    parser.add("--model", type=str, required=True)
    parser.add("--num_epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--z_type", type=str, default='none')
    parser.add("--input_context", type=str, default='both')
    #
    # # dataset
    # parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=64)
    parser.add("--n_pre_poses", type=int, default=4)
    # parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--num_workers", type=int, default=0)
    #
    # # GAN parameter
    # parser.add("--GAN_noise_size", type=int, default=0)
    #
    # # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--discriminator_lr_weight", type=float, default=0.2)
    parser.add("--loss_regression_weight", type=float, default=50)
    parser.add("--loss_gan_weight", type=float, default=1.0)
    parser.add("--loss_kld_weight", type=float, default=0.1)
    parser.add("--loss_reg_weight", type=float, default=0.01)
    parser.add("--loss_warmup", type=int, default=-1)
    #
    # # eval
    parser.add("--eval_net_path", type=str, default='')
    #
    #
    # ## add for hierachy
    # parser.add("--mse_loss_weight", type=float, default=50)
    # parser.add("--cos_loss_weight", type=float, default=50)
    # parser.add("--static_loss_weight", type=float, default=50)
    # parser.add("--motion_loss_weight", type=float, default=50)
    #
    # parser.add("--g_update_step", type=int, default=5)

    # parser.add("--AudioSize", type=int, default=128)
    # parser.add("--HiddenSize", type=int, default=300)
    # parser.add("--PoseDim", type=int, default=96)
    # parser.add("--num_epochs", type=int, default=64)
    # parser.add("--random_seed", type=int, default=-1)
    # parser.add("--model_save_path", type=str, default="/home/SENSETIME/yuzhengming/Projects/baseline/output/multi")
    # parser.add("--learning_rate", type=float, default=1e-4)
    # parser.add("--model", type=str, default="Trimodel")
    # parser.add("--name", type=str, default="Audio2Gesture")
    # parser.add("--save_result_video", type=bool, default=False)
    # # parser.add("--load_from_path", type=str, default="/mnt/lustressd/yuzhengming/gesture_data/pats/add_cyc/Audio2Gesture_checkpoint_best.bin")
    # parser.add("--load_from_path", type=str, default=None)
    # parser.add("--wordembed_dim", type=int, default=300)
    # parser.add("--wordembed_path", type=str, default='./data/fasttext/crawl-300d-2M-subword.bin') # from https://fasttext.cc/docs/en/english-vectors.html

    parser.add("--speaker", type=str, default='conan')
    parser.add("--base_path", type=str, default='')##/mnt/lustressd/yuzhengming/data/S2G
    # parser.add("--audio_flag", type=str, default='raw')
    # parser.add("--text_flag", type=str, default='bert')
    # parser.add("--comment", type=str, default='baseline')
    # parser.add("--log_dir", type=str, default='')
    parser.add("--ckpt_dir", type=str, default='')
    # parser.add("--load_from_path", type=str, default="/mnt/lustressd/yuzhengming/gesture_data/pats/add_cyc/Audio2Gesture_checkpoint_best.bin")
    parser.add('-l', "--load_from_path", type=str, default=None)
    parser.add("--save_result_video", type=str2bool, default=False)
    parser.add("--model_save_path", type=str, required=True) ##/mnt/lustressd/yuzhengming/output/base_mel512_w2v/
    parser.add("--random_seed", type=int, default=-1)

    args = parser.parse_args()

    # if len(args.comment) > 0:
    #     args.ckpt_dir = args.ckpt_dir + "_" + args.comment
    #     args.log_dir = args.log_dir + "_" + args.comment
    logging.info(args)

    return args
