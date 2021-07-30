import matplotlib.pyplot as plt
import torch

from dataset import VelaDataset
from models import AVS3Filter

# check if it is cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

# load the network
model = AVS3Filter().to(device)

# loss function, MSE(Mean Square Error) is used
loss_fun = torch.nn.MSELoss().to(device)
losses = []
losses_test = []

# load the dataset
train_set = VelaDataset(root="./", number_of_files=19140, size_of_crop=48, subset="training")
test_set = VelaDataset(root="./", number_of_files=19140, size_of_crop=48, subset="testing")
# validation_set = VelaDataset(root="./", number_of_files=19140, size_of_crop=48, subset="validation")


optimizer = torch.optim.Adam(model.parameters())
# the batch size, can be adjusted by the size of graphical memory
batch_size = 2

# dataset loader
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


# train function
def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (y, y_lr) in enumerate(train_loader):
        y, y_lr = y.to(device, torch.float), y_lr.to(device, torch.float)
        output = model(y_lr / 255)
        loss = loss_fun(output, y / 255)
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


# test function
def test(model, epoch):
    model.eval()
    for y, y_lr in test_loader:
        y_lr = y_lr.to(device, torch.float)
        y = y.to(device, torch.float)
        output = model(y_lr / 255)
        loss = loss_fun(output, y / 255)
        losses_test.append(loss)
        # 更新状态栏
        # pbar.update(pbar_update)
    print(f"\n迭代次数 {epoch}\t训练损失: {loss.item():.6f}")


if __name__ == '__main__':
    n_epoch = 100
    log_interval = 1

    # 迭代训练
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
