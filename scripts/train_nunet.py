"""NU-Net training script

This script is an example to show how to train NU-Net and to reproduce results
of the paper.
"""
import argparse
import copy
import datetime
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.onnx
import yaml
from albumentations.pytorch import ToTensorV2
from bioimageloader import Config
from bioimageloader.transforms import (BinarizeMask, ExpandToRGB,
                                       SqueezeGrayImageHWC)
from rich import traceback
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nunet import utils
from nunet.config import SelfConfig
from nunet.dataset import InfiniteIterableDataset
from nunet.transformer_net import TransformerNet
from nunet.transforms import TransformsCellpose
from nunet.vgg import Vgg19

# File provides hard-coded IDs of images whose dataset does not split
# training/testing subsets. It uses and follows bioimageloader library to load
# datasets. Find more details in the paper.
FILE_TRAINID = './config/_example/data/all_collections_trainid.yml'


def setup_logger(cmd, cfg):
    global logger
    global save_model_filename

    if cmd == 'self':
        save_model_filename = (
            f"{time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}"
            # f'_epoch{cfg.epochs:02d}'
            f'_lr{cfg.lr:.0e}'
            f'_cl{cfg.content_layer}_{cfg.content_weight:.0e}'
            f"_sl{''.join([str(n) for n in cfg.style_layer])}"
            f'_{cfg.style_weight:.0e}'
            f'_ratio{cfg.weight_ratio_style_to_content:.0f}'
            f'_br{cfg.weight_binarizer:.0e}'
            f'_rw{cfg.reg_weight:.0e}'
            f'_self_{cfg.desc}')
    elif cmd == 'sup':
        save_model_filename = (
            f"{time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}"
            f'_epoch{cfg.epochs:02d}'
            f'_sup_{cfg.desc}')
    else:
        raise NotImplementedError
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    save_log_path = os.path.join(cfg.save_model_dir,
                                 save_model_filename,
                                 f'{cmd}.log')
    check_paths(cfg)
    file_handler = logging.FileHandler(save_log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)


def printnlog(
        mesg: str,
        level=logging.INFO,
    ):
    global logger
    print(mesg)
    if level == logging.CRITICAL:
        logger.critical(mesg)
    elif level == logging.ERROR:
        logger.error(mesg)
    elif level == logging.WARNING:
        logger.warn(mesg)
    elif level == logging.INFO:
        logger.info(mesg)
    elif level == logging.DEBUG:
        logger.debug(mesg)
    else:
        raise NotImplementedError


def check_paths(cfg):
    global save_model_filename
    try:
        save_model_dir = os.path.join(cfg.save_model_dir, save_model_filename)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir, exist_ok=True)
        # if cfg.checkpoint_model_dir is not None and not (os.path.exists(cfg.checkpoint_model_dir)):
        #     os.makedirs(cfg.checkpoint_model_dir)
    except OSError as e:
        logger.error(e)
        sys.exit(1)


def _worker_init_fn(worker_id, cfg):
    global e  # epoch
    random.seed(cfg.seed + worker_id + e)
    np.random.seed(cfg.seed + worker_id + e)
    ia.random.seed(cfg.seed + worker_id + e)
    torch.manual_seed(cfg.seed + worker_id + e)


def rand_seeding(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    ia.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


def pythonize_cfg(cfg):
    cfg_dict = cfg.__dict__
    for k, v in cfg_dict.items():
        if isinstance(v, Config):
        # if isinstance(v, bioimageloader.config.Config):
            # _config_datasets_content
            # _config_datasets_style_mask
            cfg_bioimageloader = dict()
            _cfg_bioimageloader = cfg_dict[k]
            cfg_bioimageloader['filename'] = _cfg_bioimageloader.filename
            cfg_bioimageloader['datasets'] = _cfg_bioimageloader.copy()
            cfg_dict[k] = cfg_bioimageloader
    return cfg_dict


def make_checkpoint(model, history, val_history, cfg, save_model_filename,
                    n_epoch=None, n_iter=None, keep_old=False):
    """If you want to make a checkpoint during the training, it requires
    `n_epoch` and `n_iter`. Otherwise, it assumes making a checkpoint at the end
    of training by default.
    """
    root = cfg.save_model_dir
    parent = os.path.join(root, save_model_filename)
    # dir_parent = cfg.save_model_dir
    save_model_path = os.path.join(parent, 'final.model')
    if n_iter is not None and n_epoch is not None:
        # save them in a dir
        save_model_path = os.path.join(
            parent,
            f'ckpt_epoch_{n_epoch:02d}_iter_{n_iter:06d}' + '.model')
    if cfg.leave_one_out:
        # append '-{cfg._leave_one_out_dset}'
        stem, ext = os.path.splitext(save_model_path)
        save_model_path = stem + '-' + cfg._leave_one_out_dset + ext
    save_config_path = os.path.join(parent, 'config.yml')

    # model_cpu = model.cpu()  # Make a copy in CPU
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    old_models = None
    if not keep_old:
        old_models = Path(parent).glob('*.model')
        if cfg.leave_one_out:
            old_models = list(
                filter(lambda s: cfg._leave_one_out_dset in s.stem, old_models)
            )
        if n_iter is not None and n_epoch is not None:
            def _filter_old(p: Path):
                res = re.search(r'iter_\d{6}', p.stem)
                _iter = int(res.group().lstrip('iter_'))
                return _iter < n_iter
            old_models = list(filter(_filter_old, old_models))
        for p in old_models:
            logger.info(f'Delete {p}')
            os.remove(p)

    torch.save({
        'config': pythonize_cfg(cfg),
        'history': {'train': history, 'val': val_history},
        'model_path': save_model_path,
        'config_path': save_config_path,
        'state_dict': model_cpu.state_dict()
        }, save_model_path)
    logger.info(f"Save trained model at '{save_model_path}'")

    with open(save_config_path, 'w') as f:
        _config = {'Arguments': cfg.config_file,
                   'Datasets': cfg.config_datasets}
        yaml.safe_dump(_config, f, sort_keys=False)
    logger.info(f"Save training argument and config at '{save_config_path}'")

    cfg._write_saved_model_path(
        save_model_path, keep_old=keep_old, old_models=old_models
    )


def get_dataloaders(cfg):
    global logger

    print('##############################################################')
    print('    Start loading dataset')
    print('##############################################################')

    # # Define transforms
    def to_float32(img: np.ndarray, **kwargs):
        return img.astype('float32')

    # cfg._config_datasets_content.replace_commonpath('')
    # cfg._config_datasets_style_mask.replace_commonpath('')

    transforms_rotflip = [
        A.RandomRotate90(),
        A.Flip(),
    ]
    transforms_common = [
        *transforms_rotflip,
        SqueezeGrayImageHWC(),
        A.Lambda(image=to_float32, name='to_float'),
        ToTensorV2(),
    ]

    transforms_simple = A.Compose([
        A.SmallestMaxSize(256),
        A.RandomCrop(256, 256),
        *transforms_common
    ])
    transforms_cellpose = A.Compose([
        TransformsCellpose(),
        A.SmallestMaxSize(256),
        A.RandomCrop(256, 256),
        *transforms_common
    ])
    # transforms_style_mask = {
    #     'DSB2018': transforms_simple,
    #     'BBBC039': transforms_simple,
    #     'BBBC006': transforms_simple,
    # }

    transforms_foreach = {
        'DSB2018': transforms_simple,
        'ComputationalPathology': A.Compose([
            # A.SmallestMaxSize(256),
            A.RandomCrop(512, 512),
            A.Resize(256, 256),
            A.InvertImg(p=1),
            *transforms_common
        ]),
        'S_BSST265': transforms_simple,
        'BBBC006': transforms_simple,
        'BBBC039': transforms_simple,
        'Cellpose': transforms_cellpose,
        'TNBC': A.Compose([
            A.Resize(256, 256),
            A.InvertImg(p=1),
            *transforms_common
        ]),
        'BBBC007': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'DigitalPathology': A.Compose([
            A.InvertImg(p=1),
            A.RandomCrop(512, 512),
            A.Resize(256, 256),
            *transforms_common
        ]),
        'BBBC002': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC013': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC014': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC016': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC026': A.Compose([
            A.RandomCrop(512, 512),
            A.Resize(256, 256),
            *transforms_common
        ]),
        'BBBC021': A.Compose([  # Very dim
            A.RandomCrop(512, 512),
            A.Resize(256, 256),
            *transforms_common
        ]),


        # #--- abs(1.0) - ss >= 0.5 ---# #
        'FRUNet': A.Compose([
            # A.InvertImg(p=1.0), # sometimes, sometimes not
            A.SmallestMaxSize(256),  # easiest way.. no rand scaling
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC020': A.Compose([
            A.SmallestMaxSize(256),  # as well
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC008': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'MurphyLab': A.Compose([
            A.SmallestMaxSize(256),
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC018': A.Compose([
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'UCSB': A.Compose([
            A.InvertImg(p=1),
            A.RandomCrop(256, 256),
            *transforms_common
        ]),
        'BBBC015': A.Compose([
            A.RandomCrop(512, 512),  # arbitrary, still quite big
            A.Resize(256, 256),
            *transforms_common
        ]),
        'BBBC041': A.Compose([
            A.InvertImg(p=1),
            A.RandomCrop(1024, 1024),  # arbitrary
            A.Resize(256, 256),
            *transforms_common
        ]),
    }

    transforms_style_mask = A.Compose([
        ExpandToRGB(),
        A.RandomScale(0.1),
        A.SmallestMaxSize(256),
        A.RandomCrop(256, 256),
        *transforms_rotflip,
        BinarizeMask(dtype='float32', val=255.),
        ToTensorV2(transpose_mask=True),
    ])

    # Set num_samples in cfg.yml file
    # cfg_datasets_content['ComputationalPathology']['num_samples'] = 500
    datasets_content = cfg._config_datasets_content.load_datasets(
        transforms=transforms_foreach
    )
    datasets_style_mask = cfg._config_datasets_style_mask.load_datasets(
        transforms=transforms_style_mask
    )
    print('datasets_content:')
    for dset in datasets_content:
        print(f'  {dset.acronym:10s}: {len(dset):4d}')
    print('datasets_style_mask:')
    for dset in datasets_style_mask:
        print(f'  {dset.acronym:10s}: {len(dset):4d}')

    # trainid, ComPath needs training/test split
    print('datasets_content.file_list:', [len(dset.file_list) for dset in datasets_content])
    with open(FILE_TRAINID) as f:
        cfg_trainids = yaml.safe_load(f)
    datasets_content.foreach_sample_by_indices(cfg_trainids)
    print('datasets_content.file_list:', [len(dset.file_list) for dset in datasets_content])

    print('`datasets_content`')
    total_len = 0
    for dset in datasets_content:
        total_len += len(dset)
        print(f'{dset.acronym:10s}: {len(dset):10d}')
    print('{:10s}: {:10d}'.format('total', total_len))

    print('`datasets_style_mask`')
    total_len = 0
    for dset in datasets_style_mask:
        total_len += len(dset)
        print(f'{dset.acronym:10s}: {len(dset):10d}')
    print('{:10s}: {:10d}'.format('total', total_len))

    # #--- Set `num_samples` ---# #
    for dset in datasets_content:
        dset.num_samples = 36
    for dset in datasets_style_mask:
        dset.num_samples = 300

    print('`datasets_content`')
    total_len = 0
    for dset in datasets_content:
        total_len += len(dset)
        print(f'{dset.acronym:10s}: {len(dset):10d}')
    print('{:10s}: {:10d}'.format('total', total_len))

    print('`datasets_style_mask`')
    total_len = 0
    for dset in datasets_style_mask:
        total_len += len(dset)
        print(f'{dset.acronym:10s}: {len(dset):10d}')
    print('{:10s}: {:10d}'.format('total', total_len))

    cat_content = ConcatDataset(datasets_content)
    # cat_style_mask = ConcatDataset(datasets_style_mask)

    # loader_content_(train|val)
    len_val = int(len(cat_content) * cfg.validation_split)
    len_train = len(cat_content) - len_val
    datasets_content_train, datasets_content_val = random_split(
            cat_content,
            [len_train, len_val],
            generator=torch.Generator().manual_seed(cfg.seed),
    )
    loader_content_train = DataLoader(
            datasets_content_train,
            batch_size=cfg.batch_size,
            num_workers=cfg.content_workers,
            shuffle=cfg.shuffle,
            worker_init_fn=lambda x: _worker_init_fn(x, cfg),
    )
    loader_content_val = DataLoader(
            datasets_content_val,
            batch_size=cfg.batch_size,
            num_workers=cfg.content_workers,
            worker_init_fn=lambda x: _worker_init_fn(x, cfg),
            # shuffle=cfg.shuffle,
    )

    # loader_style_mask
    iterable_datasets_style_mask = InfiniteIterableDataset(*datasets_style_mask)
    loader_style_mask = DataLoader(
            iterable_datasets_style_mask,
            batch_size=cfg.style_batch_size,
            num_workers=cfg.style_workers,
            worker_init_fn=lambda x: _worker_init_fn(x, cfg),
            )
    loader_style_img = None

    return {'loader_content_train': loader_content_train,
            'loader_content_val': loader_content_val,
            'loader_style_mask': loader_style_mask,
            'loader_style_img': loader_style_img}


def balance_loss_weights(cfg, losses):
    '''Adjust `style_weight` to balance style to content weight ratio

    TODO:
        - Adjust content weight given a magnitude
        - Adjust repulsive weight as well
    '''
    global logger

    content_loss = losses['content_loss']
    attractive_loss = losses['attractive_loss']
    # repulsive_loss = losses['repulsive_loss'] # Not doing anything for now

    curr_ratio = attractive_loss / content_loss
    target_ratio = cfg.weight_ratio_style_to_content
    printnlog(f'Target weight ratio: {target_ratio}')
    printnlog(f'Balancing `style_weight` from {cfg.style_weight:.1e}')
    cfg.style_weight *= (target_ratio / curr_ratio)
    printnlog(f'                           to {cfg.style_weight:.1e}')


def self_supervise(
    cfg,
    dataloaders,
    save_model_filename,
    writer=None,
    initial_loss_run=False
):
    """Train NU-Net
    """
    global e  # epoch
    global logger

    def _get_losses(
        data: Dict[str, torch.Tensor],
        device: str,
        iter_loader_style_mask: Iterable,
        cfg: SelfConfig,
        vgg19: Vgg19,
        transformer: TransformerNet,
        # mask_transform: transforms.Compose,
        contrast_augmenters: Optional[iaa.Sequential] = None,
        weight_scheduler=None,
    ) -> dict:
        """Calculate self-supervision losses for a batch

        Parameters
        ----------
        data : torch.Tensor
            A batch to use to calculate losses
        device : str, {'cpu','cuda'}
            Use "CUDA_VISIBLE_DEVICES" environment variable to select GPU(s).

        Returns
        -------
        losses : dict
            Collection of losses ['content_loss', 'attractive_loss',
            '_attractive_loss_layers', 'repulsive_loss', 'binarizer'] in tensor
            type, except '_attractive_loss_layers'. Underscore prefix means that
            its values are not tensor type.

        """
        x = data['image']
        n_batch = x.size(0)
        # Contrast augmentation (on CPU)
        x_aug = utils.augment_batch(x,
                contrast_augmenters,
                cfg.augment_contrast_threshold) \
                if cfg.augment_contrast else x
        # transformer (NU-Net)
        x_aug = x_aug.to(device)
        y_aug = transformer(x_aug)

        # [DEPRECATED] Regularizer binarizer
        binarizer = 0.
        if cfg.weight_binarizer > 0:
            binarizer = utils.binarizer_regularizer_x2(y_aug)
            binarizer *= cfg.weight_binarizer
        # VGG
        y_aug = utils.normalize_batch(y_aug)  # Probably I shouldn't do this...
        x_aug = utils.normalize_batch(x_aug)
        features_y_aug = vgg19(y_aug, mode='feature')
        features_x_aug = vgg19(x_aug, mode='feature')
        # Content loss
        content_loss = cfg.content_weight * mse_loss(features_y_aug[cfg.content_layer],
                                                     features_x_aug[cfg.content_layer])

        # Morphological loss (a.k.a. attractive style_loss)
        attractive_loss = 0.
        _attractive_loss_layers = [0. for i in range(4)]
        features_y_aug = [features_y_aug[i] for i in cfg.style_layer]  # Filter
        grams_y_aug = [utils.gram_matrix(t) for t in features_y_aug]  # Gram(y)
        # VGG
        data_mask = next(iter_loader_style_mask)
        s_mask = data_mask['mask']
        n_sbatch = len(s_mask)
        s_mask = s_mask.to(device)
        # Attractive style features (of data['mask'])
        # s_mask = mask_transform(s_mask)
        s_mask = utils.normalize_batch(s_mask)
        features_s_image = vgg19(s_mask, mode='feature')
        features_s_image = [features_s_image[i] for i in cfg.style_layer]  # Filter
        gram_s_mask = [utils.gram_matrix(t) for t in features_s_image]
        # Attractive, manual broadcasting
        for i, batch_gm_y, sbatch_gram_mask in zip(cfg.style_layer, grams_y_aug, gram_s_mask):
            for b_ind in range(n_batch):
                gm_y = batch_gm_y[b_ind:b_ind+1, :]
                _attractive_loss = mse_loss(gm_y.repeat(n_sbatch, 1), sbatch_gram_mask)
                _attractive_loss_layers[i] += _attractive_loss.item()
                # agg_attractive_loss_layers[i] += _attractive_loss.item()
                attractive_loss += _attractive_loss
        # For loop above takes 1 from n_batch batches. Thus MSELoss does not
        # divide loss by n_batch. However, it will divide it by n_sbatch.
        # attractive_loss /= n_sbatch (wrong)
        _attractive_loss_layers = [cfg.style_weight * (v/n_batch) for v in _attractive_loss_layers]
        attractive_loss /= n_batch
        attractive_loss *= cfg.style_weight

        # [DEPRECATED] REPULSIVE style_loss (Regularizer)
        repulsive_loss = 0.
        # [DEPRECATED]
        if weight_scheduler is not None:
            content_loss = weight_scheduler.decay(content_loss)
        # Construct total loss
        total_loss = content_loss + attractive_loss + repulsive_loss + binarizer
        losses = {
            'content_loss': content_loss,
            'attractive_loss': attractive_loss,
            '_attractive_loss_layers': _attractive_loss_layers,
            'repulsive_loss': repulsive_loss,
            'binarizer': binarizer,
            'total_loss': total_loss,
        }
        return losses

    def _write_history(
        cfg: SelfConfig,
        losses: dict,
        history: Optional[dict] = None,
        agg_history: Optional[dict] = None,
    ):
        """Write loss history

        Parameters
        ----------
        cfg : SelfConfig
        losses : dict
            Collection of losses ['content_loss', 'attractive_loss',
            '_attractive_loss_layers', 'repulsive_loss', 'binarizer'] in tensor.
        history : dict, optional
            Dictionary to store losses
        agg_history : dict, optional
            Dictionary to store aggregated losses

        See Also
        --------
        _get_losses : Calculate `losses`

        """
        content_loss = losses['content_loss']
        attractive_loss = losses['attractive_loss']
        _attractive_loss_layers = losses['_attractive_loss_layers']  # not tensor
        repulsive_loss = losses['repulsive_loss']
        binarizer = losses['binarizer']
        total_loss = losses['total_loss']

        # from tensor to number
        _content_loss = content_loss.item()
        _attractive_loss = attractive_loss.item()
        # _attractive_loss_layers = _attractive_loss_layers
        _repulsive_loss = repulsive_loss.item() if cfg.reg_weight > 0 else repulsive_loss
        _binarizer = binarizer.item() if cfg.weight_binarizer > 0 else binarizer
        _total_loss = total_loss.item()

        # append to history
        if history is not None:
            history['content_loss'].append(_content_loss)
            history['attractive_loss'].append(_attractive_loss)
            history['attractive_loss_layers'].append(_attractive_loss_layers)
            history['repulsive_loss'].append(_repulsive_loss)
            history['binarizer'].append(_binarizer)
            history['total_loss'].append(_total_loss)

        # aggregate
        if agg_history is not None:
            agg_history['content_loss'] += _content_loss
            agg_history['attractive_loss'] += _attractive_loss
            agg_history['attractive_loss_layers'] = [v0+v1 for v0, v1 in zip(agg_history['attractive_loss_layers'], _attractive_loss_layers)]
            agg_history['repulsive_loss'] += _repulsive_loss
            agg_history['binarizer'] += _binarizer
            agg_history['total_loss'] += _total_loss

    print('##############################################################')
    print('    Start Training: ', save_model_filename, f'(initial_loss_run={initial_loss_run})')
    print('##############################################################')

    # [prep]
    device = torch.device("cuda" if cfg.cuda else "cpu")
    logger.debug(f'Device is set to {device}')

    loader_content_train = dataloaders['loader_content_train']
    loader_content_val = dataloaders['loader_content_val']
    if cfg.reg_weight > 0:
        loader_style_img = dataloaders['loader_style_img']
        iter_loader_style_img = iter(loader_style_img)
    loader_style_mask = dataloaders['loader_style_mask']
    iter_loader_style_mask = iter(loader_style_mask)

    # [augmenter]
    if cfg.augment_contrast:
        contrast_augmenters = iaa.Sequential([
                iaa.GammaContrast((0.5, 2.0)),
                iaa.LogContrast((0.8, 1.0)), # Larger than 1.0 may fail
                iaa.SigmoidContrast(gain=(3, 8), cutoff=(0.25, 0.75)),
            ], random_order=True)
        logger.debug(f'Contrast augmenters: {contrast_augmenters}')

    in_channels = 1
    transformer = TransformerNet(in_channels=in_channels).to(device)
    # optimizer = Adam(transformer.parameters(), cfg.lr)
    # optimizer = SGD(transformer.parameters(), cfg.lr, momentum=0.9)
    optimizer = AdamW(transformer.parameters(), cfg.lr)
    mse_loss = torch.nn.MSELoss()
    if cfg.weight_schedule:
        logger.debug('Init weight scheduler')
        weight_scheduler = utils.WeightSchedulerExp(
            gamma=cfg.weight_gamma,
            epoch_step=cfg.weight_step,
            batch_step=cfg.weight_step_batch,
            logger=logger)

    vgg19 = Vgg19(requires_grad=False).to(device)
    vgg19.eval()
    # [DEPRECATED] Centroids will be used for repulsion regularizer (repulsive_loss)
    if cfg.reg_weight > 0:
        print('Building centroids')
        sphereize = vgg19.build_centroids(
            loader_style_img,
            cfg.style_layer,
            n_iter=cfg.style_img_niter
        )
        print('[Done] Building centroids')
    else:
        _repulsive_loss = 0.0

    # [history, val_history]
    history = {
        'num_steps': 0,
        'content_loss': [],
        'attractive_loss': [],
        'attractive_loss_layers': [],
        'repulsive_loss': [],
        'binarizer': [],
        'total_loss': [],
    }
    val_history = copy.deepcopy(history)

    # [agg_history, val_agg_history]
    agg_history = {
        'content_loss': 0.,
        'attractive_loss': 0.,
        'attractive_loss_layers': [0.] * len(cfg.style_layer),
        'repulsive_loss': 0.,
        'binarizer': 0.,
        'total_loss': 0.,
    }
    _val_agg_history = copy.deepcopy(agg_history)  # template to be copied

    # [training/validation loop]
    min_val_total_loss = 1e6
    # min_val_content_loss = 1e6
    # min_val_attractive_loss = 1e6
    tolerance = 0
    for e in range(cfg.epochs):
        count = 0
        iter_loader_content_train = iter(loader_content_train)
        with tqdm(
            range(len(loader_content_train)),
            total=len(loader_content_train),
            ncols=72,
            leave=True,
            desc=f'E{e+1:02d}[TRAIN]',
        ) as pbar_train:
            for batch_id in pbar_train:
                n = batch_id + e*len(loader_content_train) + 1  # num_steps
                if cfg.weight_schedule and cfg.weight_step_batch:
                    weight_scheduler.step()
                history['num_steps'] = n
                transformer.train()
                optimizer.zero_grad()

                # [training]
                data = next(iter_loader_content_train)
                n_batch = len(data['image'])
                count += n_batch
                losses = _get_losses(
                    data=data,
                    device=device,
                    iter_loader_style_mask=iter_loader_style_mask,
                    cfg=cfg,
                    vgg19=vgg19,
                    transformer=transformer,
                    # mask_transform=mask_transform,
                    contrast_augmenters=contrast_augmenters if cfg.augment_contrast else None,
                    weight_scheduler=weight_scheduler if cfg.weight_schedule else None,
                )

                total_loss = losses['total_loss']
                total_loss.backward()
                optimizer.step()

                # [history]
                _write_history(
                    cfg,
                    losses,
                    history=history,
                    agg_history=agg_history
                )

                # [logging]
                if (batch_id + 1) % cfg.log_interval == 0:
                    # [validation]
                    n_val = len(loader_content_val)
                    # Copy the empty template
                    val_agg_history = copy.deepcopy(_val_agg_history)
                    if not initial_loss_run:
                        iter_loader_content_val = iter(loader_content_val)
                        transformer.eval()
                        # val no_grad
                        with torch.no_grad():
                            with tqdm(
                                    range(len(loader_content_val)),
                                    total=len(loader_content_val),
                                    ncols=72,
                                    leave=False,
                                    desc=f'E{e+1:02d}[VAL]',
                                ) as pbar_val:
                                for batch_id_val in pbar_val:
                                    # n_val = batch_id_val + 1
                                    # [validation]
                                    data = next(iter_loader_content_val)
                                    losses = _get_losses(
                                        data=data,
                                        device=device,
                                        iter_loader_style_mask=iter_loader_style_mask,
                                        cfg=cfg,
                                        vgg19=vgg19,
                                        transformer=transformer,
                                        # mask_transform=mask_transform,
                                        contrast_augmenters=contrast_augmenters if cfg.augment_contrast else None,
                                    )

                                    # [history]
                                    _write_history(
                                        cfg,
                                        losses,
                                        history=None,
                                        agg_history=val_agg_history,
                                    )
                        val_history['num_steps'] = n  # ??? No need, but whatever
                        for k, v in val_agg_history.items():
                            val_history[k].append(np.divide(v, n_val))

                    # [logging]
                    mesg = ("\n"
                        "{}\tEpoch {}:\t"
                        "[{}/{}]\t"
                        "content: {:.4f}\t ({:.4f})"
                        "attraction: {:.4f}\t ({:.4f})"
                        # "repulsion: {:.2f}\t"
                        # "binarizer: {:.2f}\t"
                        "total: {:.4f}\t ({:.4f})").format(
                            time.ctime(), e + 1,
                            count, len(loader_content_train.dataset),
                            agg_history['content_loss'] / n, val_agg_history['content_loss'] / n_val,
                            agg_history['attractive_loss'] / n, val_agg_history['attractive_loss'] / n_val,
                            # agg_repulsive_loss / n,
                            # agg_binarizer / n,
                            agg_history['total_loss'] / n, val_agg_history['total_loss'] / n_val,
                    )
                    # print(mesg)
                    pbar_train.write(mesg)
                    logger.info(mesg)

                    # TensorBoard
                    if writer:
                        writer.add_scalar('content_loss', agg_history['content_loss'] / n, n)
                        writer.add_scalar('attractive_loss', agg_history['attractive_loss'] / n, n)
                        for i in range(len(cfg.style_layer)):
                            writer.add_scalar(f'attractive_loss_layers[{i}]', agg_history['attractive_loss_layers'][i] / n, n)
                        writer.add_scalar('repulsive_loss', agg_history['repulsive_loss'] / n, n)
                        writer.add_scalar('binarizer', agg_history['binarizer'] / n, n)
                        writer.add_scalar('total_loss', agg_history['total_loss'] / n, n)

                        writer.add_scalar('val_content_loss', val_agg_history['content_loss'] / n_val, n)
                        writer.add_scalar('val_attractive_loss', val_agg_history['attractive_loss'] / n_val, n)
                        for i in range(len(cfg.style_layer)):
                            writer.add_scalar(f'val_attractive_loss_layers[{i}]', val_agg_history['attractive_loss_layers'][i] / n_val, n)
                        writer.add_scalar('val_repulsive_loss', val_agg_history['repulsive_loss'] / n_val, n)
                        writer.add_scalar('val_binarizer', val_agg_history['binarizer'] / n_val, n)
                        writer.add_scalar('val_total_loss', val_agg_history['total_loss'] / n_val, n)

                    # [early stopping]
                    if not initial_loss_run:
                        val_total_loss = val_agg_history['total_loss'] / n_val
                        # val_content_loss = val_agg_history['content_loss'] / n_val
                        # val_attractive_loss = val_agg_history['attractive_loss'] / n_val
                        if val_total_loss < min_val_total_loss:
                            min_val_total_loss = val_total_loss
                            tolerance = 0
                            mesg = (
                                "{}\tEpoch {}:\tn={:4d}\t"
                                "`val_total_loss` improved, making a checkpoint and "
                                "reset tolerance to 0").format(
                                    time.ctime(), e + 1, n,
                            )
                            pbar_train.write(mesg)
                            logger.info(mesg)
                            make_checkpoint(
                                transformer,
                                history,
                                val_history,
                                cfg,
                                save_model_filename,
                                n_epoch=e + 1,
                                n_iter=n,
                                keep_old=cfg.keep_checkpoints)
                        else:
                            tolerance += 1
                            mesg = (
                                "{}\tEpoch {}:\tn={:4d}\t"
                                "min_val_total_loss: {:.4f}\t"
                                "val_total_loss: {:.4f}\t"
                                "tolerance: {:2d}").format(
                                    time.ctime(), e + 1, n,
                                    min_val_total_loss,
                                    val_total_loss,
                                    tolerance,
                            )
                            pbar_train.write(mesg)
                            logger.info(mesg)
                            if cfg.early_stopping and tolerance >= cfg.tolerance:
                                pbar_train.write(f"Early Stopping (n={n})")
                                return
        if initial_loss_run:
            # After the first epoch
            del transformer
            del vgg19
            return {
                'content_loss': agg_history['content_loss'] / n,
                'attractive_loss': agg_history['attractive_loss'] / n,
                'repulsive_loss': agg_history['repulsive_loss'] / n,
            }
        if cfg.weight_schedule and cfg.weight_step and not cfg.weight_step_batch:
            weight_scheduler.step()

    # Save the final model
    make_checkpoint(
        transformer,
        history,
        val_history,
        cfg,
        save_model_filename,
        n_epoch=e + 1,
        n_iter=n,
        keep_old=cfg.keep_checkpoints)


def supervise(cfg, dataloaders, save_model_filename, writer=None):
    ...


def stylize(args):
    ...


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main(cmd, cfg):
    global logger
    global save_model_filename

    if not torch.cuda.is_available():
        logger.warn("CUDA is not available")

    # check_paths(cfg)
    rand_seeding(cfg)
    if cfg.leave_one_out:
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.log_dir,
                save_model_filename + '-' + cfg._leave_one_out_dset
            )
        )
    else:
        writer = SummaryWriter(
            log_dir=os.path.join(cfg.log_dir,
                                 save_model_filename)
        )

    if cmd == 'self':
        logger.info('Perform self-supervision')
        dataloaders = get_dataloaders(cfg)
        # Test training to balance losses
        if cfg.weight_ratio_style_to_content > 0:
            losses = self_supervise(cfg, dataloaders, 'TestRun', initial_loss_run=True)
            balance_loss_weights(cfg, losses) # Adjust `style_weight`
        self_supervise(cfg, dataloaders, save_model_filename, writer=writer)
    # elif cmd == 'sup':
    #     logger.info('Perform supervision')
    #     dataloaders = get_dataloaders_sup(cfg)
    #     supervise(cfg, dataloaders, save_model_filename, writer=writer)
    # elif cmd == 'eval':
    #     # stylize(args)
        # raise NotImplementedError
    else:
        raise NotImplementedError
    writer.close()


def parse_args():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser(
        "self",
        help="parser for self-supervision training arguments")
    train_arg_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="config file for self-supervision")
    args = main_arg_parser.parse_args()
    return args


def parse_config(args):
    cmd = args.subcommand
    if cmd == 'self':
        cfg = SelfConfig(args.config)
    # elif cmd == 'sup':
    #     cfg = SupConfig(args.config)
    # elif cmd == 'eval':
    #     raise NotImplementedError
    else:
        raise NotImplementedError
    return cmd, cfg


if __name__ == '__main__':
    # global
    e = 0  # epoch
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    save_model_filename = ''

    traceback.install()  # prettify traceback
    args = parse_args()
    cmd, cfg = parse_config(args)
    setup_logger(cmd, cfg)
    printnlog(f'config: {cfg.config_file}\n{cfg}')

    # Set _config_dataset
    cfg_datasets_content = Config(cfg.config_datasets['content'])
    cfg_datasets_style_mask = Config(cfg.config_datasets['style_mask'])
    setattr(cfg, '_config_datasets_content', cfg_datasets_content)
    setattr(cfg, '_config_datasets_style_mask', cfg_datasets_style_mask)

    tic = time.time()
    main(cmd, cfg)
    toc = time.time()

    t = str(datetime.timedelta(seconds=toc-tic))
    printnlog(f'Executed in {t}')
