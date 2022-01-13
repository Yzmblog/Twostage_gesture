import argparse
from tqdm import tqdm
import subprocess
import os
import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
# parser.add_argument('-output_path', '--output_path', default='output directory to save cropped intervals', required=True)
# parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

# args = parser.parse_args()

base_path = "/media/SENSETIME\yuzhengming/DATA/PATS"
output_path = "/media/SENSETIME\yuzhengming/DATA/PATS/crop_intervals"
speaker = False

# base_path = "/mnt/lustre/share/yuzhengming/pats/"
# output_path = "/mnt/lustrenew/yuzhengming/pats/crop_intervals"
# speaker = False


s2g_speakers = ['almaram', 'angelica', 'chemistry', 'conan', 'ellen', 'jon', 'oliver', 'rock', 'seth', 'shelly']

def save_interval(input_fn, start, end, output_fn):
    cmd = 'ffmpeg -i "%s" -ss %s -to %s -strict -2 "%s" -y' % (
    input_fn, start, end, output_fn)
    subprocess.call(cmd, shell=True, stdin=subprocess.DEVNULL)

if __name__ == "__main__":
    missing_vedio = dict()
    missing_num = 0
    df_intervals = pd.read_csv(os.path.join(base_path, 'all', 'data', 'cmu_intervals_df.csv'))
    if speaker:
        df_intervals = df_intervals[df_intervals["speaker"] == speaker]

    for _, interval in tqdm(df_intervals.iterrows(), total=len(df_intervals)):
        if interval['speaker'] != 'ferguson':
            continue
        try:
            start_time = str(pd.to_datetime(interval['start_time']).time())
            end_time = str(pd.to_datetime(interval['end_time']).time())
            input_fn = os.path.join(base_path, "videos", interval['speaker'], "videos", interval["video_fn"])
            output_fn = os.path.join(output_path, interval["speaker"], "%s_%s_%s-%s.mp4"%(interval["interval_id"], interval["video_fn"], str(start_time), str(end_time)))
            if os.path.exists(output_fn):
                continue
            if os.path.exists(input_fn):
                print(input_fn, output_fn)
                out_dir = os.path.join(output_path, interval["speaker"])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                save_interval(input_fn, str(start_time), str(end_time), output_fn)
            else:
                if missing_vedio.__contains__(interval['speaker']):
                    missing_vedio[interval['speaker']].add(interval["video_fn"])
                else:
                    missing_vedio[interval['speaker']] = set()
                    missing_vedio[interval['speaker']].add(interval["video_fn"])
                missing_num += 1
                print(input_fn, " didn't exist, could not be cropped", " totally missed:", missing_num)

        
        except Exception as e:
            print(e)
            print("couldn't crop interval: %s"%interval)
