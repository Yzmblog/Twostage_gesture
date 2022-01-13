# import numpy as np
# with open('/media/SENSETIME\yuzhengming/DATA/video_data/Chemistry/train/filepaths.npy') as f:
#     data = np.load(f)
#     print(type(data))
from model.tcn import TemporalConvNet
import torch
# tcn = TemporalConvNet(num_inputs=6, num_channels=[6, 8])
# test_tensor = torch.randn((6, 6, 8))
# print(tcn(test_tensor).shape)


money = [188, 234, 241, 276, 556, 561]
res = []
minmun_mon = 1e9
def dfs(sum, iter):
    global minmun_mon
    if iter == 6:
        if sum < minmun_mon and sum >= 1200:
            minmun_mon = sum
            print('minum_mon now:', minmun_mon, res)
            return
        else:
            return
    if sum > 1300 or sum > minmun_mon:
        return
    dfs(sum, iter=iter + 1)
    num = money[iter]
    res.append(num)
    dfs(sum + num, iter + 1)
    del res[-1]

if __name__ == '__main__':
    dfs(0, 0)
    print(res)


