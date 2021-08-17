import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image, ImageReadMode, write_png

from models import AVS3Filter

# check if it is cuda available
# if 'cuda out of memory' is threw, try to set device as "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    # Here you can adjust the num_workers, number of cpu cores are recommended
    num_workers = 4
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

# load the network
net_filename = './Trained_Networks/256.pkl'
model = torch.load(net_filename, map_location=device)


# transform the output(float tensor) to a unsigned int8 tensor,
# and deal with the overflow of values(x >255 or x < 0)# manually
def clear_overflow(tensor):
    zero_value = torch.tensor([0]).to(device, torch.float)
    neg_clear = torch.heaviside(tensor, zero_value)
    result_tensor = tensor * neg_clear + 0.01
    overflow = result_tensor - 255
    pos_clear = torch.heaviside(overflow, zero_value) * overflow
    result_tensor = result_tensor - pos_clear
    result_tensor = result_tensor.floor()
    result_tensor = result_tensor.squeeze(0)
    result_tensor = result_tensor.to(device, torch.uint8)
    return result_tensor


im_filename = '10.jpg'
# read the testing image file
im = read_image(im_filename, mode=ImageReadMode.GRAY)
im = im.to(device, torch.float) / 255
# output
out = model(im.unsqueeze(0))
out = clear_overflow(out * 255)
# write back to disk
write_png(out.cpu(), 'out_' + im_filename)
out = out.permute(1, 2, 0)
plt.imshow(out.numpy(), cmap='gray')
plt.show()
