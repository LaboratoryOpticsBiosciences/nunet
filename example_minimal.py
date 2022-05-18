"""Run NU-Net

1. Download 'config/' and 'models_filter'
2. pip install -e .
    'nunet' module will be available
"""

import re
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms

from nunet.config import Config, SelfConfig
from nunet.transformer_net import TransformerNet


def load_checkpoints(cfg: Config) -> List[TransformerNet]:
    style_models = []
    for p in cfg._saved_model_path:
        model = torch.load(p)
        state_dict = model['state_dict']
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model = TransformerNet()
        style_model.load_state_dict(state_dict)
        style_models.append(style_model)
    return style_models


def load_model(
    cfg: SelfConfig,
    ind: int = -1
):
    print('len(cfg._saved_model_path):', len(cfg._saved_model_path))
    models = load_checkpoints(cfg)

    common_path = Path(cfg._saved_model_path[0]).parent.stem
    print('common_path:', {common_path})
    models_names = [Path(p).stem for p in cfg._saved_model_path]

    for i, name in enumerate(models_names):
        print(f'{i:3d}, {name}')

    model_name = models_names[ind]
    print(f'Loading {ind}: {model_name}')
    nu_net = models[ind]
    nu_net.cuda()
    nu_net.eval()
    return common_path, model_name, nu_net


def run_nu_net(img, nu_net) -> np.ndarray:
    with torch.no_grad():
        tensor = numpy2torch(img).cuda()
        out_tensor = nu_net(tensor)
        out_tensor_clipped = torch.clip(out_tensor, 0, 255)
        out_np_clipped = torch2numpy(out_tensor_clipped)
    return out_np_clipped / 255.0


def numpy2torch(
    array: np.ndarray,
    device: Optional[str] = None,
    cuda: Optional[bool] = None,
) -> torch.Tensor:
    """Cast an image array to a tensor, ready to be consumed by NU-Net

    Parameters
    ----------
    array : numpy.ndarray
        A numpy array
    device : Optional
        If given, cast the tensor to the specified device. Argument to
        ``torch.Tensor.to()``
    cuda : Optional
        Copy the tensor from CPU to GPU. Ignored if `device` is given.

    Returns
    -------
    torch.Tensor
        Data range is assumed to be UINT8 but the actual dtype will be FLOAT32
        for direct computation.

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    tensor = transform(array)
    tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    elif cuda is not None:
        tensor = tensor.cuda()
    return tensor


def torch2numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """Cast pytorch tensor(s) to numpy array(s)

    """
    # (1,c,y,x)
    ch = tensor.size(1)
    tensor = tensor.detach().cpu().squeeze()
    array = tensor.permute(1, 2, 0).numpy() if ch == 3 else tensor.numpy()
    return array


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
