from typing import Optional

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import CenterCrop

"""
    Author : Vela Yang
    Last edited : 30th, July, 2021
    Framework : PyTorch
    This .py file is the Vela Dataset class, based on 'iaprtc12' dataset( http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz )
    It includes 19140 high resolution images and the corresponding low resolution images
    Designed for image restoration use
"""
# known files with error, should be excluded
error_filename = [9659, 10549, 10552]


class VelaDataset(Dataset):

    """
    _walker : the list includes number markers of image files
    _path : the directory of the dataset
    _crop : the crop transform to images
    """
    _walker = None
    _path = None
    _crop = None

    def __init__(self,
                 root,
                 number_of_files=19140,
                 size_of_crop=None,
                 subset: Optional[str] = None):

        # default crop size(no crop)
        if size_of_crop is None:
            size_of_crop = [360, 480]

        # root directory
        self._path = root

        # crop function
        #self._crop = CenterCrop(size=size_of_crop)

        # divide the dataset
        assert subset is None or subset in ["training", "validation", "testing"], (
                "When `subset` not None, it must take a value from "
                + "{'training', 'validation', 'testing'}."
        )

        # divide the dataset, training for 60%, testing for 20% and validation for 20%
        if subset == "training":
            self._walker = range(1, int(number_of_files / 5 * 3))
        elif subset == "validation":
            self._walker = range(int(number_of_files / 5 * 3), int(number_of_files / 5 * 4))
        elif subset == "testing":
            self._walker = range(int(number_of_files / 5 * 4), number_of_files)
        self._walker = [i for i in self._walker if i not in error_filename]

    # implement the __getitem__ method
    def __getitem__(self, n: int):

        # excludes the files with error
        if n in error_filename:
            n = 1

        # get the file name
        filename = self._walker[n]

        # load image
        image = read_image(self._path + "HighRes/" + str(filename) + ".jpg", mode=ImageReadMode.GRAY)
        image_lr = read_image(self._path + "LowRes/" + str(filename) + ".jpg", mode=ImageReadMode.GRAY)


        # if the image should be transposed
        shape = image.shape
        shape_lr = image_lr.shape
        if shape[1] > 400:
            image = image.permute(0, 2, 1)
        if shape_lr[1] > 400:
            image_lr = image_lr.permute(0, 2, 1)
        return image, image_lr

    def __len__(self):
        return len(self._walker)
