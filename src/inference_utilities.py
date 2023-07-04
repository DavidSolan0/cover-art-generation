# Disable pylint error E1101 for the entire code
# pylint: disable=E1101
# Disable pylint error E1102 for the entire code
# pylint: disable=E1102

import torch
from diffusers import StableDiffusionPipeline
from torch.cuda import autocast


def generate_images(
    prompt,
    negative_prompt,
    num_samples,
    guidance_scale,
    num_inference_steps,
    height,
    width,
):
    weights_path = (
        "artifacts/conditioned/tunned"  # Specify the path to the weights file
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 123  # Specify the desired seed value

    # Load the diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        weights_path, torch_dtype=torch.float16
    ).to(device)

    g_cuda = torch.Generator(device=device)
    g_cuda.manual_seed(SEED)

    with autocast(device), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda,
        ).images

    return images
