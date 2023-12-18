import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np

def show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img[0][0])
        plt.axis('off')
    plt.savefig('mnist.png')

def load_transformed_dataset(dataset_name, img_size=64):
    data_transform = transforms.Compose([
        transforms. Resize((img_size, img_size)), 
        transforms. ToTensor(), # Scales data into [0,1]
        transforms. Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST
    else:
        raise ValueError

    train = dataset(root=".", download=True, transform=data_transform, train=True)
    test = dataset(root=".", download=True, transform=data_transform, train=False)    
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image, filename, savefig=False, channel_last=False, is_jnp_array=False):

    if is_jnp_array:
        image = torch.from_numpy(np.array(image))
    
    if channel_last: # Make B, C, H, W 
        image = image.permute(0, 3, 1, 2)

    image = image.cpu()
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), #CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    if savefig:
        plt.savefig(filename)
