#!/usr/bin/env python3
#https://huggingface.co/docs/diffusers/quicktour

import PIL.Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler #EulerDiscreteScheduler, UNet2DModel, DDPMScheduler
import tqdm
import torch
import sys

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to(device)


t_range=[0.02, 0.98]
pipe.num_train_timesteps = scheduler.config.num_train_timesteps
pipe.min_step = int(pipe.num_train_timesteps * t_range[0])
pipe.max_step = int(pipe.num_train_timesteps * t_range[1])
pipe.alphas = scheduler.alphas_cumprod.to(device) # for convenience


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]


inputs = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length, return_tensors='pt')
embeddings = pipe.text_encoder(inputs.input_ids.to(pipe.device))[0]

image.save("astronaut_rides_horse.png")


scheduler.set_timesteps(50)

latents = torch.randn((embeddings.shape[0] // 2, pipe.unet.in_channels, 512 // 8, 512 // 8), device)

for i, t in enumerate(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    # predict the noise residual
    noise_pred = pipe.unet(latent_model_input, t, embeddings)['sample']

    # perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents)['prev_sample']




#def produce_latents(text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

#    if latents is None:
#        latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)





#print(embeddings.shape[0] // 2, pipe.unet.in_channels, 512 // 8, 512 // 8)


#latents = torch.randn((embeddings.shape[0] // 2, pipe.unet.in_channels, 512 // 8, 512 // 8), x)

#sys.exit('ggggggggggggggggggggggggggggggggggggggggggg')





print('===================== INPUTS ==============================')
print(inputs)
print('===================== EMBEDDINGS ==============================')
print(embeddings)
print('===================== VAE ==============================')
print(pipe.vae)
print('================== TOKENIZER =============================')
print(pipe.tokenizer)
print('================== TEXT_ENCODER =============================')
print(pipe.text_encoder)
print('================== UNET =============================')
print(pipe.unet)






