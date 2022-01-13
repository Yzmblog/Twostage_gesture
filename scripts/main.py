import datetime
import logging
import torch

import sys

[sys.path.append(i) for i in ['.', '..']] ### Add search path of 'import'

from config.parse_args import parse_args
from data import Dataset


if __name__ == '__main__':
    args = parse_args()
    dataset = Dataset(args, 'train')
    # print(dataset.__len__())#85964
    dataset_dev = Dataset(args, 'dev')
    # print(dataset_dev.__len__())# 9713
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    for i, batch in enumerate(dataloader):
        if i > 0:
            break
        print('audio shape: ', batch['audio'].shape,
              'text shape: ', batch['text'].shape,
              'pose shape: ', batch['pose'].shape)
