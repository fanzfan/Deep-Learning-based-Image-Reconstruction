import os
from random import random
from typing import Optional

import torch
import numpy as np
from torch import Tensor
import cv2

from torch.utils.data import Dataset

error_filename = [9659, 10549, 10552]

class MyDataset(Dataset):
    def __init__(self,
                 root,
                 number_of_files=19140,
                 subset: Optional[str] = None):
        assert subset is None or subset in ["training", "validation", "testing"], (
                "When `subset` not None, it must take a value from "
                + "{'training', 'validation', 'testing'}."
        )
        self._path = root

        if subset == "training":
            self._walker = range(1, int(number_of_files / 5 * 3))
        elif subset == "validation":
            self._walker = range(int(number_of_files / 5 * 3), int(number_of_files / 5 * 4))
        elif subset == "testing":
            self._walker = range(int(number_of_files / 5 * 4), number_of_files)
        self._walker = [i for i in self._walker if i not in error_filename]

    def __getitem__(self, n: int) -> tuple[Tensor, Tensor]:
        if n in error_filename:
            n = 1
        filename = self._walker[n]
        image = cv2.imread(self._path + "HighRes/" + str(filename)+ ".jpg")
        image_lr = cv2.imread(self._path + "LowRes/" + str(filename) + ".jpg")
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        ycrcb_lr = cv2.cvtColor(image_lr, cv2.COLOR_RGB2YCR_CB)
        y = ycrcb[:, :, 0]
        y_lr = ycrcb_lr[:, :, 0]
        y = y if np.shape(y)[0] == 480 else y.T
        y_lr = y_lr if np.shape(y_lr)[0] == 480 else y_lr.T
        if np.shape(y_lr)[0] == 360:
            print("Here")
        return torch.from_numpy(y).unsqueeze(0), torch.from_numpy(y_lr).unsqueeze(0)

    def __len__(self):
        return len(self._walker)
