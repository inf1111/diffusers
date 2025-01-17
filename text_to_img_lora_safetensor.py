#!/usr/bin/env python3
#https://github.com/huggingface/diffusers/issues/2829
#https://github.com/huggingface/diffusers/issues/3064
#чето еще про конвертацию 3 форматов которые принимает проект:
#https://www.reddit.com/r/StableDiffusion/comments/10h2ltj/diffusers_ckpt_and_safetensors/
#https://github.com/haofanwang/Lora-for-Diffusers/blob/18adfa4da0afec46679eb567d5a3690fd6a4ce9c/format_convert.py#L154-L161
#https://github.com/huggingface/diffusers/issues/3064
#https://github.com/huggingface/diffusers/issues/2829
#https://github.com/huggingface/diffusers/issues/2105

from diffusers import StableDiffusionPipeline
import torch

import safetensors
from collections import defaultdict
from diffusers.loaders import LoraLoaderMixin

def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = safetensors.torch.load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

pipe = load_lora_weights(pipe, "/home/ubuntu/diffusers/edgMarquise.safetensors", 1.0, "cuda", torch.float32)

imgs = pipe(prompt = "A girl with blue eyes EDGMARQUISE HAIRSTYLE", num_images_per_prompt=5).images;

for index, item in enumerate(imgs):
   item.save(f"SIMPLE-{index}.png")