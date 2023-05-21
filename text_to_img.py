#!/usr/bin/env python3
#https://huggingface.co/docs/diffusers/quicktour

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
image = pipeline("A pokemon with blue eyes.").images[0]
image.save("pokemon5.png")