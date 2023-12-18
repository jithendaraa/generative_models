from flax import linen as nn


from jax import numpy as jnp
from jax import jit
from unet import UNet


class MNISTDiffusionJAX(nn.Module):
    image_size: int
    in_channels: int
    betas: jnp.array
    alphas: jnp.array
    alphas_cumprod: jnp.array
    sqrt_alphas_cumprod: jnp.array
    sqrt_one_minus_alphas_cumprod: jnp.array
    time_embedding_dim: int = 256
    timesteps: int = 1000
    base_dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)

    def setup(self):
        self.model = UNet(
            self.timesteps, 
            self.time_embedding_dim,
            self.in_channels,
            self.in_channels,
            self.base_dim,
            self.dim_mults
        )

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        b = x_0.shape[0]
        sqrt_alphas_cumprod_t = jnp.take_along_axis(self.sqrt_alphas_cumprod, t, axis=-1).reshape(b, 1, 1, 1)
        sqrt_alphas_cumprod_tp1 = jnp.take_along_axis(self.sqrt_one_minus_alphas_cumprod, t, axis=-1).reshape(b, 1, 1, 1)
        out = sqrt_alphas_cumprod_t * x_0 + sqrt_alphas_cumprod_tp1 * noise
        return out

    def __call__(self, x_t, t, train):
        pred_noise = self.model(x_t, t, train)
        return pred_noise

    

    def _reverse_diffusion(self, pred, x_t, t, noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        b = x_t.shape[0]
        alpha_t = jnp.take_along_axis(self.alphas, t, axis=-1).reshape(b, 1, 1, 1)
        alpha_t_cumprod = jnp.take_along_axis(self.alphas_cumprod, t, axis=-1).reshape(b, 1, 1, 1)
        beta_t = jnp.take_along_axis(self.betas, t, axis=-1).reshape(b, 1, 1, 1)
        
        sqrt_one_minus_alpha_cumprod_t = jnp.take_along_axis(self.sqrt_one_minus_alphas_cumprod, t, axis=-1).reshape(b, 1, 1, 1)
        mean = (1./jnp.sqrt(alpha_t)) * (x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t) * pred)

        if t.min() > 0:
            alpha_t_cumprod_prev = jnp.take_along_axis(self.alphas_cumprod, t-1, axis=-1).reshape(b, 1, 1, 1)
            std = jnp.sqrt(beta_t * (1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean + std * noise


    
    