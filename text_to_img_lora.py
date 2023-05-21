#!/usr/bin/env python3
#https://huggingface.co/docs/diffusers/v0.13.0/en/training/lora

from diffusers import StableDiffusionPipeline
import torch

model_path = "sayakpaul/sd-model-finetuned-lora-t4" # в этой строке вся разница
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path) # в этой строке вся разница
pipe.to("cuda")

prompt = "A pokemon with blue eyes."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon-xxx-5.png")