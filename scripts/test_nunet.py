"""Run NU-Net given a checkpoint and an input image

1. Download 'nunet_models.tar' on GitHub and unzip it in the root directory
2. Run `pip install -e .`
3. Run this script from the root directory
    See help message of this script by executing `python scripts/test_nunet.py --help`
"""

import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nunet.config import SelfConfig
from nunet.utils import load_model, numpy2torch, torch2numpy


def run_nu_net(img, nu_net, cuda=True) -> np.ndarray:
    img = check_input_channel(img)
    with torch.inference_mode():
        tensor = numpy2torch(img)
        if cuda:
            tensor = tensor.cuda()
        out_tensor = nu_net(tensor)
        out_tensor_clipped = torch.clip(out_tensor, 0, 255)
        out_np_clipped = torch2numpy(out_tensor_clipped)
    return out_np_clipped / 255.0


def check_input_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        if img.ndim == 3:
            # Assume all the channels are the same and return the first channel
            return img[..., 0]
        else:
            raise NotImplementedError
    return img


def check_outdir(filename: Path):
    if filename.exists():
        return
    else:
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
            print('Made parent directories')
    return


def check_outext(filename: Path):
    # default suffix is .png
    if not filename.suffix:
        return filename.with_suffix('.png'), '.png'
    return filename, filename.suffix


def main(args):
    # #--- Load model ---# #
    cfg = SelfConfig(args.cfg)
    common_path, model_name, nu_net = load_model(cfg,
                                                 ind=args.id,
                                                 cuda=args.cuda)

    # #--- Load image ---# #
    img = np.array(Image.open(args.infile))

    # #--- Run model ---# #
    img_out = run_nu_net(img, nu_net, cuda=args.cuda)

    # #--- Save outfile ---# #
    check_outdir(args.outfile)
    outfile, ext = check_outext(args.outfile)
    if ext.lower() == '.tif':
        outimg = Image.fromarray(img_out)
        outimg.save(outfile)
    elif ext.lower() == '.png' or ext.lower() == '.jpg':
        img_out = 255 * img_out
        outimg = Image.fromarray(img_out)
        outimg = outimg.convert('L')
        outimg.save(outfile)
    else:
        raise NotImplementedError('Supported file extension: .tif, .png, .jpg')


if __name__ == '__main__':
    parser = ArgumentParser(description="Test a NU-Net given an image")
    parser.add_argument('cfg', type=Path,
                        help="Config for a model")
    parser.add_argument('infile', type=Path,
                        help="Input image")
    parser.add_argument('outfile', type=Path,
                        help="Output filename")
    parser.add_argument('-i', '--id', type=int, default=-1,
                        help="Choose a checkpoint by its id. -1 means the "
                        "last one. (default: -1)")
    parser.add_argument('--cuda', action='store_true',
                        help="Use CUDA device")
    args = parser.parse_args()

    t0 = time.time()
    main(args)
    t1 = time.time()
    print(f'Executed in {(t1 - t0) / 60:.2f} minutes')
