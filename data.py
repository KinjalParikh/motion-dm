from torch.utils.data import DataLoader
import numpy as np
import os
import torch


class HumanAct12Poses(torch.utils.data.Dataset):
    dataname = "humanact12"

    def __init__(self, datapath="dataset", **kargs):
        super().__init__(**kargs)
        datafilepath = os.path.join(datapath, "HumanAct12_unconstrained.npy")
        self._data = np.load(datafilepath)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, ind):
        data = self._data[ind]
        centered = data - data[0, 0, :]  # first frame's first joint's xyz position
        return torch.from_numpy(centered).permute(1, 2, 0)  # model expects njoints X nfeats X nframes


def get_dataset_loader(batch_size):
    dataset = HumanAct12Poses()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True
    )
    return loader