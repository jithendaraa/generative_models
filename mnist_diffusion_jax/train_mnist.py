import os
from tqdm import tqdm
import math
import numpy as np

import jax
from jax import numpy as jnp
from jax import jit
import optax
import wandb

from utils import ( 
    create_mnist_dataloaders, parse_args, TrainState, 
    update_key_and_noise, train_step, _reverse_diffusion_with_clip
)

from model import MNISTDiffusionJAX
import torch
from torchvision.utils import save_image

_HALF_PI = math.pi*0.5
WANDB_PROJECT = "diffusion"
WAND_ENTITY = "disentangle_diffusion"


def _cosine_variance_schedule(timesteps, epsilon= 0.008):
    steps = jnp.linspace(0, timesteps, timesteps+1, dtype=jnp.float32)
    value = ((steps/timesteps+epsilon)/(1.0+epsilon)) * _HALF_PI
    f_t = jnp.cos(value)**2
    betas = jnp.clip(1.0 - f_t[1:]/f_t[:timesteps], 0.0, 0.999)
    return betas


def main(args):
    if not args.off_wandb:  
        wandb.init(
            project = WANDB_PROJECT, 
            entity = WAND_ENTITY, 
            config = vars(args), 
            settings = wandb.Settings(start_method="fork")
        )

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size,
        image_size=28
    )

    key = jax.random.PRNGKey(args.seed)
    fixed_key = jax.random.PRNGKey(args.seed)

    betas = _cosine_variance_schedule(args.timesteps)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=-1)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - alphas_cumprod)

    diffuser = MNISTDiffusionJAX(
        image_size=28,
        in_channels=1, 
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod, 
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        time_embedding_dim=256, 
        timesteps=args.timesteps,
        base_dim=args.model_base_dim,
        dim_mults=(2,4),
    )

    forward_diffusion = jit(diffuser._forward_diffusion)
    image_batch = jnp.array(next(iter(train_dataloader))[0].numpy())
    image_batch = jnp.transpose(image_batch, (0, 2, 3, 1))
    print(f"Batched Image shape: {image_batch.shape}")
    noise = jax.random.normal(key, image_batch.shape)

    dummy_t = jax.random.randint(key, (args.batch_size,), 0, diffuser.timesteps)
    x_t = forward_diffusion(image_batch, dummy_t, noise)
    params = diffuser.init(key, x_t, dummy_t, True)
    params, batch_stats = params['params'], params['batch_stats']

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Number of params: {param_count}")
    print()
    
    state = TrainState.create(
        apply_fn=diffuser.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.adamw(args.lr),
    )

    assert args.no_clip is False

    def sampling(key, state, n_samples):
        x_t = jax.random.normal(key, shape=(n_samples, diffuser.image_size, diffuser.image_size, diffuser.in_channels))
        for i in tqdm(range(diffuser.timesteps-1, -1, -1),desc="Sampling"):
            key, _ = jax.random.split(key, 2)
            t = jnp.full(shape=(n_samples,), fill_value=i, dtype=jnp.int32)
            sample1, sample2 = _reverse_diffusion_with_clip(key, state, x_t, t, noise, alphas, alphas_cumprod, betas)

            if t.min() > 0: x_t = sample1
            else:           x_t = sample2

        x_t = (x_t + 1.) / 2. #[-1,1] to [0,1]
        return x_t

    global_steps = 0
    wandb_counter = 0

    for i in range(args.epochs):
        for j, (image, _) in enumerate(tqdm(train_dataloader)):
            image = jnp.transpose(image.cpu().numpy(), (0, 2, 3, 1)) # BHWC
            b = image.shape[0]
            key, noise = update_key_and_noise(key, image)
            
            t = jax.random.randint(key, (b,), 0, diffuser.timesteps)
            x_t = forward_diffusion(image, t, noise)
            loss, state = train_step(state, x_t, t, noise)
            global_steps+=1

            if (j+1)%50 == 0:
                samples = sampling(key, state, args.n_samples)
                fixed_samples = sampling(fixed_key, state, args.n_samples)
                sample_filename = plot_samples(samples, global_steps, args.n_samples, False)
                fixed_sample_filename = plot_samples(fixed_samples, global_steps, args.n_samples, True)        

                if args.off_wandb is False:
                    wandb.log({
                        'loss': loss,
                        'Generated Samples/Random': wandb.Image(sample_filename),
                        'Generated Samples/Fixed': wandb.Image(fixed_sample_filename),
                    }, step=global_steps)

        print(f"Epoch {i} | Loss: {loss:.3f}")


def plot_samples(samples, global_steps, n_samples, fixed=False):
    samples = jnp.transpose(samples, (0, 3, 1, 2)) # BCHW
    samples = torch.tensor(np.array(samples))

    if fixed:
        dirname = "fixed_results"
    else:
        dirname = "results"

    os.makedirs(dirname, exist_ok=True)
    filename = f"{dirname}/" + "steps_{:0>8}.png".format(global_steps)
    save_image(samples, filename, nrow=int(math.sqrt(n_samples)))
    print(f"Saved at {filename}!")
    return filename

if __name__=="__main__":
    args = parse_args()
    main(args)


