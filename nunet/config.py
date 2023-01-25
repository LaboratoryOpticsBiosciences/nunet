import logging
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


class Config:
    """Base config class"""
    # #--- Global ---# #
    # Description
    #  (REQUIRED)
    desc: str = ''

    # True to use GPU, False to use CPU
    #  Default: True
    cuda: bool = True

    def __repr__(self):
        out = '\n'
        def _filter(a):
            return (not a.startswith('_') and
                    not callable(self.__getattribute__(a)))
        attr = filter(_filter, dir(self))
        return out.join([f'{k} : {self.__getattribute__(k)}' for k in attr])

    def __init__(self, config_file: str, zip_file: Optional[zipfile.ZipFile]=None):
        """Override config

        Parameters
        ----------
        config_file : str
            Path to a config file
        """
        self.config_file = config_file

        if zip_file is not None:
            with zip_file.open(config_file) as f:
                cfg = yaml.safe_load(f)
        else:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
        no_matches = []
        results = {}
        overridden = {}
        for k, v in cfg.items():
            if k.startswith('_'):
                # Underscore keys are reserved for results
                setattr(self, k, v)
                results[k] = v
            else:
                if hasattr(self, k):
                    setattr(self, k, v)
                    overridden[k] = v
                else:
                    no_matches.append(k)

        if no_matches:
            # not_matched_attr = filter(lambda x: not x.startswith('_'), no_matches)
            for k in no_matches:
                logger.error(f"'{config_file}':Config key '{k}' does not exist")
            self._no_matches = no_matches
            sys.exit(1)

        if results:
            logger.warn(f"'{config_file}':This config file was already used")
            self._results = results

        if overridden:
            self._overridden = overridden

        self._check_required('desc')

    def _check_required(self, *attr):
        empty = 0
        for a in attr:
            # if not getattr(self, a):
            if not self.__getattribute__(a):
                empty += 1
                logger.error(f"'{self.config_file}':'{a}' is empty")
        if empty > 0:
            sys.exit(1)

    def _write_saved_model_path(
        self,
        path: str,
        keep_old=False,
        old_models=None
    ):
        """Make key '_saved_model_path' if not existed and append `path` to it.
        """
        # with open(self.config_file, 'r') as f:
        #     cfg = yaml.safe_load(f)

        with open(self.config_file, 'rt') as f:
            lines = f.readlines()

        new_lines = []
        if (s := '_saved_model_path:\n') not in lines:
        # if '_saved_model_path' not in lines[0]:
        # if '_saved_model_path' not in cfg.keys():
            new_lines.append('_saved_model_path:\n')
            new_lines.append(f'  - {path}\n')
            new_lines.append('\n')
            new_lines.extend(lines)
        else:
            ind0 = lines.index(s)
            new_lines.extend(lines)  # copy
            count = lines[ind0:].index('\n')
            new_lines.insert(ind0 + count, f'  - {path}\n')
            if not keep_old:
                if old_models is not None:
                    for om in old_models:
                        new_lines.remove(f'  - {om}\n')
                else:
                    for _ in range(count - 1):
                        new_lines.pop(ind0 + 1)
        with open(self.config_file, 'w') as f:
            f.writelines(new_lines)

    @classmethod
    def glob_configs(
        cls,
        directory: str,
        key: Optional[Callable]=None
    ):
        """Glob '{directory}/*.yml' and return Configs

        Parameters
        ----------
        directory : str
            Path to directory to load
        key : callable
            Sorting key func that takes Config and output boolean

        Examples
        --------
        >>> cfgs = Config.glob_configs('./config/experiment/test_exp/',
                                       key=lambda a: a.lr)
        """
        root = Path(directory)
        assert root.exists() and root.is_dir()

        _cfgs = sorted(root.glob('*.yml'))
        cfgs = [cls(p) for p in _cfgs]
        if key:
            cfgs = sorted(cfgs, key=key)
        return cfgs


class SelfConfig(Config):
    """Self-supervision config"""
    # Path to folder where trained model will be saved
    #  (REQUIRED)
    save_model_dir: str = ''

    # #--- Dataloader ---# #
    # Config file to load datasets
    #  (REQUIRED)
    # new ver. using bioimageloader
    config_datasets: Dict[str, str] = {
        'content': '',
        'style_mask': '',
    }
    # config_datasets: str = ''

    # True to perform Leave-One-dataset-Out experiment
    #  Default: False
    leave_one_out: bool = False

    # Validation split ratio
    #  Default: 0.2
    validation_split: float = 0.2

    # Batch size for content dataloader
    #  Default: 16
    batch_size: int = 16

    # Number of workers for content dataloader
    #  Default: 2
    content_workers: int = 2

    # Batch size for style dataloader
    #  Default: 512
    style_batch_size: int = 512

    # Number of workers for style dataloader
    #  Default: 8
    style_workers: int = 8

    # #--- Data augumentation ---# #
    # Random seed
    #  Default: 42
    seed: int = 42

    # Shuffle dataset
    #  Default: True
    shuffle: bool = True

    # Perform contrast augumentation on content images
    #  Default: True
    augment_contrast: bool = True

    # Minimum max-min contrast requirement, negative value to ignore threshold
    #  Default: 16 (uint8)
    augment_contrast_threshold: int = 16


    # #--- Training ---# #
    # Number of training epochs
    #  Default: 2
    epochs: int = 2

    # Learning rate
    #  Default: 1e-3
    lr: float = 1e-3

    # Early stopping
    #  Default: True
    early_stopping: bool = True

    # Keep or remove previous checkpoints
    #  Default: False
    keep_checkpoints: bool = False

    # How many times to tolerate increased validation loss for early stopping.
    # If `early_stopping` is False, it will be ignored.
    #  Default: 10
    tolerance: int = 10

    # Feature layer index of MobileNetV2 for content loss
    #  Default: 1 (0-index)
    content_layer: int = 1

    # Feature layer index of MobileNetV2 for style loss
    #  Default: [0,1,2,3,4,5,6] (0-index)
    style_layer: List[int] = list(range(7))

    # Balance the weight ratio by running a few batches and by adjusting
    # `style_weight` according to the given ratio. NOTE that the actual ratio
    # will never be accurate. If < 0, it is disabled.
    #  Default: 3.0
    weight_ratio_style_to_content: float = 3.0

    # Weight for content loss
    #  Default: 1e1
    content_weight: float = 1e1

    # Weight scheduling (currently only for content_weight)
    #  Default: False
    weight_schedule: bool = False

    # Weight decay gamma
    #  Default: 0.9
    weight_gamma: float = 0.9

    # Weight scheduling interval. Only one of `weight_step` or
    # `weight_step_epoch` will be accepted. (unit: epoch iteration number)
    #  Default: 1
    weight_step: int = 1

    # Weight scheduling interval. Only one of `weight_step` or
    # `weight_step_epoch` will be accepted. If given, `weight_step` will be
    # ignored. (unit: batch iteration number)
    #  Default: None
    weight_step_batch: Optional[int] = None

    # Weight for style loss
    #  Default: 1e4
    style_weight: float = 1e4


    # #--- Log ---# #
    # Path to tensorboard log directory
    #  Default: 'runs/'
    log_dir: str = 'runs/'

    # Number of iterations after which the loss is logged
    #  Default: 10
    log_interval: int = 10

    # Number of iterations after which predicted images is logged
    #  Default: 20
    log_image_interval: int = 20


    # #--- Optional ---# #
    # Path to folder where checkpoints of trained models will be saved
    #  (OPTIONAL)
    checkpoint_model_dir: Optional[str] = None

    # Number of iterations after which a checkpoint of the trained model will be
    # created
    #  Default: 2000
    checkpoint_interval: int = 2000


    # #--- Deprecated ---# #
    # Weight for repulsion regularizer. When <= 0, it will be disabled.
    #  Default: -1.0
    reg_weight: float = -1.0

    # Weight for binarizer regularizer. When <= 0, it will be disabled.
    #  Default: -1.0
    weight_binarizer: float = -1.0

    def __init__(self, config_file: str, zip_file: Optional[zipfile.ZipFile]=None):
        super().__init__(config_file, zip_file)
        self._check_required('save_model_dir', 'config_datasets')



class SupConfig(Config):
    """Supervision config"""
    # Path to an initial model (possibly pre-trained one) to begin training with
    #  (OPTIONAL)
    init_model: Optional[str] = None

    # Path to folder where trained model will be saved
    #  (REQUIRED)
    save_model_dir: str

    # #--- Dataloader ---# #
    # Config file to load datasets
    #  (REQUIRED)
    config_datasets: str

    # Batch size for dataloader
    #  Default: 32
    batch_size: int = 32

    # Number of workers for dataloader
    #  Default: 4
    num_workers: int = 4


    # #--- Data augumentation ---# #
    # Random seed
    #  Default: 42
    seed: int = 42

    # Shuffle dataset
    #  Default: True
    shuffle: bool = True

    # Perform contrast augumentation on content images
    #  Default: True
    augment_contrast: bool = True

    # Minimum max-min contrast requirement, negative value to ignore threshold
    #  Default: 16 (uint8)
    augment_contrast_threshold: int = 16


    # #--- Training ---# #
    # Number of training epochs
    #  Default: 2
    epochs: int = 2

    # Learning rate
    #  Default: 1e-3
    lr: float = 1e-3


    # #--- Log ---# #
    # Path to tensorboard log directory
    #  Default: 'runs/'
    log_dir: str = 'runs/'

    # Number of iterations after which the loss is logged
    #  Default: 10
    log_interval: int = 10


    # #--- Optional ---# #
    # Path to folder where checkpoints of trained models will be saved
    #  (OPTIONAL)
    checkpoint_model_dir: str = ''

    # Number of iterations after which a checkpoint of the trained model will be
    # created
    #  Default: 2000
    checkpoint_interval: int = 2000

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self._check_required('save_model_dir', 'config_datasets')
