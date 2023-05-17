#!/usr/bin/env python3
#https://huggingface.co/docs/diffusers/quicktour

import PIL.Image
import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler
import tqdm

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id)
scheduler = DDPMScheduler.from_config(repo_id)

def save_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save(f"img{i}.png")
    print(f"Image at step {i}")
    #display(image_pil)

torch.manual_seed(0)

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
torch.Size([1, 3, 256, 256])

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample

model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        save_sample(sample, i + 1)