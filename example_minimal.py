"""Run NU-Net

1. Download 'config/' and 'models_filter'
2. pip install -e .
    'nunet' module will be available
"""

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from nunet.config import Config, SelfConfig
from nunet.transformer_net import TransformerNet
from nunet.utils import load_checkpoints, load_model, numpy2torch, torch2numpy


def run_nu_net(img, nu_net) -> np.ndarray:
    with torch.no_grad():
        tensor = numpy2torch(img).cuda()
        out_tensor = nu_net(tensor)
        out_tensor_clipped = torch.clip(out_tensor, 0, 255)
        out_np_clipped = torch2numpy(out_tensor_clipped)
    return out_np_clipped / 255.0


def main(args):
    # #--- Load model ---# #
    cfg = SelfConfig(args.cfg)
    common_path, model_name, nu_net = load_model(cfg)

    # #--- Load image ---# #
    ...

    # #--- Run model ---# #
    img_out = run_nu_net(img, nu_net)


if __name__ == '__main__':
    parser = ArgumentParser(description="Plot results of ultimate(!) NU-Net")
    parser.add_argument('cfg', type=Path, help='Config for a model')
    # parser.add_argument('-i', '--id', type=int, nargs='+',
    #                     help='Choose model(s) by their ids. -1 means the '
    #                     'latest one. Cannot be used with --all.')
    # parser.add_argument('-a', '--all', action='store_true',
    #                     help='Choose all models. Cannot be used with --id.')
    args = parser.parse_args()
    # if args.all:
    #     assert args.id is None, "Cannot give `id` when `all` is given"

    t0 = time.time()
    main(args)
    t1 = time.time()
    print(f'Executed in {(t1 - t0) / 60:.2f} minutes')
