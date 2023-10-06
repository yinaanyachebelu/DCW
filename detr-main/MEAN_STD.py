import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

data_path = '/home/ayina/MscThesis/DCW/datasets/Dataset_final/DATA_mean_std'

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform_img
)

batch_size = 4

loader = DataLoader(
    image_data,
    batch_size=batch_size,
    num_workers=1)


def batch_mean_and_sd(loader):

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
            cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)
    return mean, std


if __name__ == '__main__':
    mean, std = batch_mean_and_sd(loader)
    print("mean and std: \n", mean, std)
