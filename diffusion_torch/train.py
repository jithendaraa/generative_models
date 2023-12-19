import os
import argparse
import logging

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from utils.helper import show_images, load_transformed_dataset, show_tensor_image

from models.UNet import SimpleUNet
from models.Diffusion import Diffusion


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def main(args):
    IMG_SIZE = 64
    BASEDIR = "/home/mila/j/jithendaraa.subramanian/scratch"
    WANDB_PROJECT = "disentangle_diffusion"
    WAND_ENTITY = "jithendaraa"
    CHANNELS = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_transformed_dataset(args.data, img_size=IMG_SIZE)
    show_images(data, num_samples=20, cols=4)

    if args.off_wandb is False:  
        wandb.init(
            project = WANDB_PROJECT, 
            entity = WAND_ENTITY, 
            config = vars(args), 
            settings = wandb.Settings(start_method="fork")
        )

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SimpleUNet(CHANNELS).to(device)
    diffuser = Diffusion(
        noise_steps=args.diffusion_timesteps, 
        beta_start=1e-4,
        beta_end=0.02,
        img_size=IMG_SIZE, 
        device=device
    )
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")
    image = next(iter(dataloader))[0]
    plt.figure(figsize=(15, 5))
    num_images = 10
    stepsize = int(args.diffusion_timesteps / num_images)

    for idx in tqdm(range(0, args.diffusion_timesteps, stepsize), desc='Generating forward diffusion samples..'):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, (idx // stepsize) + 1)
        plt.axis('off')
        image, noise = diffuser.forward_diffusion_sample(image, t)
        savefig = (idx == args.diffusion_timesteps - stepsize)
        show_tensor_image(image, f"diffused_{idx}.png", savefig)

    @torch.no_grad()
    def sample_plot_image(device, timesteps, figname, img_size=IMG_SIZE):
        img = torch.randn(1, CHANNELS, img_size, img_size).to(device)
        plt.figure(figsize=(15, 5))
        num_images = 10
        stepsize = int(timesteps / num_images)
        for i in tqdm(range(0, timesteps)[::-1], desc='Reverse diffusion'):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = diffuser.reverse_diffusion_sample(model, img, t)
            img = torch.clamp(img, -1.0, 1.0)
            filename = f'{figname}_reverse_diffusion_t{i}.png'
            savefig = (i == 0)

            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i / stepsize) + 1)
                show_tensor_image(img.detach().cpu(), filename, savefig=savefig)

    def get_loss(model, x0, t):
        x_noisy, noise = diffuser.forward_diffusion_sample(x0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    optimizer = Adam(model.parameters(), lr=args.lr)

    start = 0
    if args.load_checkpoint != '':
        checkpoint_num = int(args.load_checkpoint)
        start = checkpoint_num + 1
        filename = f"{BASEDIR}/diffusion_mnist_epoch{checkpoint_num}.pt"
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model params from {filename}")

    for epoch in range(start, args.epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            gt_image = batch[0].to(device)
            label = batch[1].to(device)
            if gt_image.shape[1] == 1:
                gt_image = gt_image.expand(-1, CHANNELS, -1, -1)
            t = torch.randint(0, args.diffusion_timesteps, (args.batch_size, ), device=device).long()
            loss = get_loss(model, gt_image, t)
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(device, args.diffusion_timesteps, figname=f"epoch{epoch}_step{step}", img_size=IMG_SIZE)

        if epoch % 20 == 0:
            print("Saving model params")
            torch.save(model.state_dict(), f"{BASEDIR}/diffusion_mnist_epoch{epoch}.pt")
            print("saved model params")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--diffusion_timesteps', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--off_wandb', type=bool, default=True)
    args = parser.parse_args()
    main(args)