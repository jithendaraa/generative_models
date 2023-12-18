import torch
import torch.nn.functional as F
import logging

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(device) # beta_{1:T}
        self.alphas = 1. - self.betas # (T, )

        # \bar{alpha}_t = alphas_cumprod_prev[t] = prod_{s=1}^{t} alpha_s = prod_{s=1}^{t} (1 - beta_s) for t = 1...T
        # \bar{alpha}_0 = 1
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(self.device) # (T, )
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.).to(self.device) # (T, )

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_1m_alpha_bar = torch.sqrt(1. - self.alpha_bar)
        self.posterior_variances = self.betas * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar) 

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        shape = ((1, ) * (len(x_shape) - 1))
        out = out.reshape(batch_size, *shape).to(self.device)
        return out

    def forward_diffusion_sample(self, x0, t):
        """
            x_0: (B, C, H, W)
            t: int
        """
        t = t.to(self.device)
        x0 = x0.to(self.device)
        sqrt_alpha_bar_t = self.get_index_from_list(self.sqrt_alpha_bar, t, x0.shape).to(self.device)
        sqrt_1m_alpha_bar_t = self.get_index_from_list(self.sqrt_1m_alpha_bar, t, x0.shape).to(self.device)
        noise = torch.randn_like(x0).to(self.device)
        x_t = sqrt_alpha_bar_t * x0 + sqrt_1m_alpha_bar_t * noise # Reparam sample from N(sqrt_alpha_bar_t * x0, sqrt_1m_alpha_bar_t ** 2 I)
        return x_t, noise
    
    def reverse_diffusion_sample(self, model, x, t):
        x_shape = x.shape
        sqrt_alpha_t = torch.sqrt(self.get_index_from_list(self.alphas, t, x_shape))
        beta_t = self.get_index_from_list(self.betas, t, x_shape)
        sqrt_one_minus_alpha_bar_t = self.get_index_from_list(self.sqrt_1m_alpha_bar, t, x_shape)

        model_mean = (x - (beta_t * model(x, t) / sqrt_one_minus_alpha_bar_t)) / sqrt_alpha_t
        posterior_variance_t = self.get_index_from_list(self.posterior_variances, t, x_shape)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x) #  N(0, I)
            posterior_scale = torch.sqrt(posterior_variance_t)
            return model_mean + posterior_scale * noise