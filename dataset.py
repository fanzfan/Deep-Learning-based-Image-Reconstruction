from typing import Optional
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import CenterCrop

# 目前已知的，读取有问题的样本
error_filename = [9659, 10549, 10552]
# 让输入图片等维
transform = CenterCrop(size=48)

class VelaDataset(Dataset):
    def __init__(self,
                 root,
                 number_of_files=19140,
                 subset: Optional[str] = None):
        # 划分数据集
        assert subset is None or subset in ["training", "validation", "testing"], (
                "When `subset` not None, it must take a value from "
                + "{'training', 'validation', 'testing'}."
        )
        self._path = root

        # 划分数据集，其中训练集60%，测试集20%，验证集20%
        if subset == "training":
            self._walker = range(1, int(number_of_files / 5 * 3))
        elif subset == "validation":
            self._walker = range(int(number_of_files / 5 * 3), int(number_of_files / 5 * 4))
        elif subset == "testing":
            self._walker = range(int(number_of_files / 5 * 4), number_of_files)
        self._walker = [i for i in self._walker if i not in error_filename]

    # 实现getitem方法
    def __getitem__(self, n: int):
        if n in error_filename:
            n = 1
        filename = self._walker[n]
        image = read_image(self._path + "HighRes/" + str(filename) + ".jpg", mode=ImageReadMode.GRAY)
        image_lr = read_image(self._path + "LowRes/" + str(filename) + ".jpg", mode=ImageReadMode.GRAY)
        """
        shape = image.shape
        shape_lr = image_lr.shape
        if shape[1] > 400:
            image = image.permute(0, 2, 1)
        if shape_lr[1] > 400:
            image_lr = image_lr.permute(0, 2, 1)
        """
        return transform(image), transform(image_lr)

    def __len__(self):
        return len(self._walker)
