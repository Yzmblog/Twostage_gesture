import sys
import os
from tqdm import tqdm
import subprocess
import argparse



def main(speaker):
    ## remote
    base_path = '/mnt/lustressd/yuzhengming/data/pats/crop_intervals'
    result_path = '/mnt/lustressd/yuzhengming/data/pats/crop_frames'
    ##local
    # base_path = '/media/SENSETIME\\yuzhengming/DATA/PATS/crop_intervals'
    # result_path = '/media/SENSETIME\\yuzhengming/DATA/PATS/crop_frames'
    base_path = os.path.join(base_path, speaker)
    result_path = os.path.join(result_path, speaker)


    all_intervals = os.listdir(base_path)

    for interval in all_intervals:
        id = interval.split('_')[0]
        interval_path = os.path.join(base_path, interval)
        final_path = os.path.join(result_path, id)
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        cmd = 'ffmpeg -i %s %s' % (interval_path, final_path) + '/frame_%d.jpg'
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", help="speaker name",  type=str)
    args = parser.parse_args()
    main(args.speaker)
