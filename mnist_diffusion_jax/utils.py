import argparse
import os
import pickle

from torchvision import transforms 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import jax
from jax import jit, value_and_grad
from jax import numpy as jnp
import optax
from flax.training import train_state
from typing import Any



def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    preprocess=transforms.Compose([
        transforms.Resize(image_size),\
        transforms.ToTensor(),\
        transforms.Normalize([0.5],[0.5])
    ]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    test_dataset=MNIST(root="./mnist_data", train=False, download=True, transform=preprocess)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return train_dataloader, test_dataloader
            

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int ,default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path',default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet',default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM',default=300)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay',default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence',default=10)
    parser.add_argument('--no_clip', action='store_true', help='set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--off_wandb', action='store_true')
    args = parser.parse_args()
    return args


class TrainState(train_state.TrainState):
  batch_stats: Any = None


@jit
def update_key_and_noise(key, image):
    key, _ = jax.random.split(key, 2)
    noise = jax.random.normal(key, image.shape)
    return key, noise


@jit
def train_step(state: TrainState, x_t, t, noise):
    def loss_fn(params, batch_stats):
        pred, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats}, 
            x_t, 
            t, 
            train=True, 
            mutable=['batch_stats']
        )
        loss = optax.l2_loss(pred, noise).mean()
        return loss, updates

    loss, _ = loss_fn(state.params, state.batch_stats)
    grad_fn = value_and_grad(loss_fn, argnums=(0), has_aux=True)
    (loss, updates), grads = grad_fn(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return loss, state


@jit
def _reverse_diffusion_with_clip(key, state, x_t, t, noise, alphas, alphas_cumprod, betas):
    b = x_t.shape[0]
    noise = jax.random.normal(key, x_t.shape)
    pred = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x_t, 
        t,
        train=False
    )

    alpha_t = jnp.take_along_axis(alphas, t, axis=-1).reshape(b, 1, 1, 1)
    alpha_t_cumprod = jnp.take_along_axis(alphas_cumprod, t, axis=-1).reshape(b, 1, 1, 1)
    beta_t = jnp.take_along_axis(betas, t, axis=-1).reshape(b, 1, 1, 1)
    
    x_0_pred = jnp.sqrt(1. / alpha_t_cumprod) * x_t - jnp.sqrt(1. / alpha_t_cumprod - 1.) * pred
    x_0_pred = jnp.clip(x_0_pred, -1., 1.)

    alpha_t_cumprod_prev = jnp.take_along_axis(alphas_cumprod, t-1, axis=-1).reshape(b, 1, 1, 1)
    mean1 = (beta_t * jnp.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred +\
            ((1. - alpha_t_cumprod_prev) * jnp.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t
    std1 = jnp.sqrt(beta_t * (1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
    
    mean2 = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
    
    sample1 = mean1 + std1 * noise 
    sample2 = mean2
    return sample1, sample2
