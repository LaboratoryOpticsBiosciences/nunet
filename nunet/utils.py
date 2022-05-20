import os
import re
from pathlib import Path, PureWindowsPath
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms

from .config import Config, SelfConfig
from .transformer_net import TransformerNet


def to_legal_ospath(path: str) -> str:
    """Detect the OS and replace any eventual illegal path character with "_".
    This will allow file transfer without filename check from Linux to Windows.

    Parameters
    ----------
    path: str
        The input path

    Returns
    -------
    legal_path: str
        A path with no illegal windows character but in posix path format
    """
    if os.name == "nt":  # If running on Windows
        # Replace all illegal characters with underscore
        legal_path = re.sub(r"\*|:|<|>|\?|\|", '_', path)
        # Turn the Windows path into a Posix path if it isn't already
        legal_path = PureWindowsPath(legal_path).as_posix()
    elif os.name == "posix":  # If running Linux or MacOS
        legal_path = PureWindowsPath(legal_path).as_posix()
    return(legal_path)


def load_checkpoints(cfg: Config) -> List[TransformerNet]:
    style_models = []
    for p in cfg._saved_model_path:
        model = torch.load(to_legal_ospath(p))
        state_dict = model['state_dict']
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model = TransformerNet()
        style_model.load_state_dict(state_dict)
        style_models.append(style_model)
    return style_models


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
