from random import random

import torch

from models import AVS3Filter
from dataset import MyDataset
import torch.nn.functional as F
from torchvision.io import read_image, encode_jpeg, write_jpeg
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
def read(path: str):
    image = cv2.imread(path)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    y = ycrcb[:, :, 0]
    ycrcb[:, :, 1] = 128
    ycrcb[:, :, 2] = 128
    im = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
    plt.imshow(im)
    plt.show()
    im2 = cv2.imread('./1.jpg')
    ycrcb = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)
    ycrcb[:, :, 0] = y
    x = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
    cv2.imwrite('2.jpg', x, [cv2.IMWRITE_PNG_COMPRESSION, 90])
    a = 1
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False
model = AVS3Filter()
model.to(device)
mse_loss = torch.nn.MSELoss()
train_set = MyDataset(root="./", number_of_files=19140, subset="training")
test_set = MyDataset(root="./", number_of_files=19140, subset="testing")
optimizer = torch.optim.Adam(model.parameters())

batch_size = 256

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)
losses = []
losses_test = []


# optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 40次迭代后开始降低学习率，避免过拟合

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, images in enumerate(train_loader):
        y, y_lr = images
        output = model(y_lr)
        loss = F.nll_loss(output.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 展示训练状态
        if batch_idx % log_interval == 0:
            print(
                f"迭代次数: {epoch} [{batch_idx * len(y_lr)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t训练损失: {loss.item():.6f}")
        # 更新状态栏
        # pbar.update(pbar_update)
        # 记录训练损失
        losses.append(loss.item())





def test(model, epoch):
    model.eval()
    for y, y_lr in test_loader:
        y_lr = y_lr.to(device)
        y = y.to(device)
        output = model(y_lr)
        loss = F.nll_loss(output.squeeze(), y)
        losses_test.append(loss)
    print(f"\n迭代次数 {epoch}\t训练损失: {loss.item():.6f}")


if __name__ == '__main__':
    n_epoch = 100
    log_interval = 20
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)

    # 展示训练损失与准确率
    plt.subplot(2, 1, 1)
    plt.plot(losses);
    plt.title("训练损失");
    plt.subplot(2, 1, 2)
    plt.plot(losses_test);
    plt.title("测试损失");
