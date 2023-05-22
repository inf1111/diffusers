#!/usr/bin/env python3
#https://huggingface.co/docs/diffusers/quicktour

from diffusers import DiffusionPipeline

#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = DiffusionPipeline.from_pretrained("/home/ubuntu/diffusers/conv/2")
pipeline.to("cuda")

imgs = pipeline(prompt = "An image of a lkjh ring in Picasso style", num_images_per_prompt=5).images;

for index, item in enumerate(imgs):
   item.save(f"Picasso-{index}.png")
