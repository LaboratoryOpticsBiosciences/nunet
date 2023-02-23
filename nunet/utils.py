import os
import re
import zipfile
from pathlib import Path, PureWindowsPath
from typing import List, Optional

import imgaug.augmenters as iaa
import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import Config, SelfConfig
from .transformer_net import TransformerNet


def load_models_from_zipfile(
    f: str,
    default_model_config='config/_example/exp/filter_slider',
):
    zip_file = zipfile.ZipFile(f)
    def _filter_cfg(i: zipfile.ZipInfo) -> bool:
        return not i.is_dir()
    cfgs = list(filter(
        lambda zi: re.search(rf'{default_model_config}.+', zi.filename),
        zip_file.filelist
    ))
    cfgs = list(filter(_filter_cfg, cfgs))

    def _get_sw(s: str) -> Optional[int]:
        res = re.search(r'sw\d+', s)
        if res is not None:
            r = res.group()
            return int(r.lstrip('sw'))
        return None

    def _sort_cfgs(zi: zipfile.ZipInfo):
        fname: str = zi.filename
        return _get_sw(fname)
    cfgs_sorted = sorted(cfgs, key=_sort_cfgs)

    self_cfgs = [SelfConfig(_, zip_file=zip_file) for _ in cfgs_sorted]
    models = dict()
    for cfg in self_cfgs:
        common_path, model_name, nu_net = load_model(cfg, cuda=False,
                                                     zip_file=zip_file)
        models[_get_sw(common_path)] = nu_net
    return models


def load_checkpoints(
    cfg: Config,
    zip_file: Optional[zipfile.ZipFile] = None,
) -> List[TransformerNet]:
    style_models = []
    for p in cfg._saved_model_path:
        if zip_file is not None:
            with zip_file.open(p) as f:
                model = torch.load(f)
        else:
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
    ind: int = -1,
    cuda: bool = True,
    zip_file: Optional[zipfile.ZipFile] = None,
):
    print('len(cfg._saved_model_path):', len(cfg._saved_model_path))
    models = load_checkpoints(cfg, zip_file)

    common_path = Path(cfg._saved_model_path[0]).parent.stem
    print('common_path:', {common_path})
    models_names = [Path(p).stem for p in cfg._saved_model_path]

    for i, name in enumerate(models_names):
        print(f'{i:3d}, {name}')

    model_name = models_names[ind]
    print(f'Loading {ind}: {model_name}')
    nu_net = models[ind]
    if cuda:
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


def match_out_shape(
    out_img: np.ndarray,
    in_img: np.ndarray
):
    """Sometimes the output shape does not match the input shape

    Parameters
    ----------
    out_img : numpy.ndarray
        (Y, X)
    in_img : numpy.ndarray
        (Y, X)
    """
    in_shape = in_img.shape
    x = in_shape[1]
    y = in_shape[0]
    return cv2.resize(out_img, (x, y))


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram.view(b, ch * ch)


def normalize_batch(batch: torch.Tensor, n_channels=3) -> torch.Tensor:
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # size=(3,1,1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    if n_channels == 1:
        mean = mean.mean(0, keepdim=True)
        std = std.mean(0, keepdim=True)
    batch = batch.div(255.0)
    return (batch - mean) / std


def augment_batch(
    batch: torch.Tensor,
    augmenters: iaa.Sequential,
    threshold: int,
    loop_limit: int = 10
) -> torch.Tensor:
    """Contrast augmentation

    Parameters
    ----------
    batch : torch.Tensor
        Batch of tensors whose shape is (b,1,h,w)
    augmenters : imgaug.augmenters.Sequential
        Sequence of augmenters
    threshold : uint8 or int < 0
        Ensure minimum contrast
    loop_limit : int
        Limit number of loop

    Returns
    -------
    augmented : torch.Tensor
        Tensor having shape of (batch, channel=1, height, width)

    Notes
    -----
    - Only support grayscale image tensor (tensor.size(1)==1)

    """
    assert batch.size(1) == 1, 'Not supported tensor shape'
    images = batch.permute(0, 2, 3, 1).squeeze(dim=-1).numpy().astype(np.uint8)
    augmented = []
    for img in images:
        aug = augmenters.augment_image(img)  #***
        n = 1
        if threshold < 0:
            aug = torch.tensor(aug[np.newaxis, :, :], dtype=batch.dtype)
            augmented.append(aug)
            continue
        while aug.max() - aug.min() < threshold:
            if n >= loop_limit:
                aug = img
                break
            aug = augmenters.augment_image(img)
            n += 1
        aug = torch.tensor(aug[np.newaxis, :, :], dtype=batch.dtype)
        augmented.append(aug)
    return torch.stack(augmented, dim=0)


# def to_legal_ospath(path: str) -> str:
#     """Detect the OS and replace any eventual illegal path character with "_".
#     This will allow file transfer without filename check from Linux to Windows.

#     Parameters
#     ----------
#     path: str
#         The input path

#     Returns
#     -------
#     legal_path: str
#         A path with no illegal windows character but in posix path format
#     """
#     if os.name == "nt":  # If running on Windows
#         # Replace all illegal characters with underscore
#         legal_path = re.sub(r"\*|:|<|>|\?|\|", '_', path)
#         # Turn the Windows path into a Posix path if it isn't already
#         legal_path = PureWindowsPath(legal_path).as_posix()
#     elif os.name == "posix":  # If running Linux or MacOS
#         legal_path = PureWindowsPath(legal_path).as_posix()
#     return legal_path


# def load_checkpoints(
#     cfg: Config,
#     zip_file: Optional[zipfile.ZipFile] = None,
# ) -> List[TransformerNet]:
#     style_models = []
#     for p in cfg._saved_model_path:
#         if zip_file is not None:
#             with zip_file.open(p) as f:
#                 model = torch.load(f)
#         else:
#             model = torch.load(to_legal_ospath(p))
#         state_dict = model['state_dict']
#         for k in list(state_dict.keys()):
#             if re.search(r'in\d+\.running_(mean|var)$', k):
#                 del state_dict[k]
#         style_model = TransformerNet()
#         style_model.load_state_dict(state_dict)
#         style_models.append(style_model)
#     return style_models
