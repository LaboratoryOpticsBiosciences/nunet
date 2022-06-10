import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class InfiniteIterableDataset(IterableDataset):
    """Wrapper to enable infinite sampling from the given datasets, respecting
    batch functionality of torch.utils.DataLoader

    Examples
    --------
    # Prepare datasets
    dataset1 = Dataset(...)
    dataset2 = Dataset(...)
    dataset3 = Dataset(...)
    # Wrap it
    iterable_dataset = InfiniteIterableDataset(dataset1, dataset2, dataset3)
    loader_iterable_dataset = DataLoader(iterable_dataset, batch_size=32)
    iter_infinite = iter(loader_iterable_dataset)
    # Call a sample
    batch = next(iter_infinite)
    """
    def __init__(self, *datasets):
        """
        Parameters
        ----------
        datasets : torch.utils.data.Dataset
        """
        super().__init__()
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.len_datasets = [len(d) for d in datasets]
        self.cumsum_len = np.cumsum(self.len_datasets)

    def __len__(self):
        """This length is not the length of iteration, but the (either real or
        virtual) total number of datasets
        """
        return self.cumsum_len[-1]

    def _call(self):
        ind = random.randint(0, len(self)-1)
        # Find in which dataset ind is in
        a = 0
        for i_dataset, b in enumerate(self.cumsum_len):
            if ind < b:
                ind -= a
                break
            a = b
        return self.datasets[i_dataset][ind]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return iter(self._call, None)
        else:
            worker_id = worker_info.id
            # print('id =', worker_id)
        return iter(self._call, None)
