# Disable pylint error E1101 for the entire code
# pylint: disable=E1101
# Disable pylint error E1102 for the entire code
# pylint: disable=E1102

import sys
import argparse

import torch
import torchvision.utils as vutils
from torch import autocast
from diffusers import StableDiffusionPipeline
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt

from models import ContextUnet, Generator
from diffusion_utilities import plot_sample, sample_ddpm, sample_ddim

parser = argparse.ArgumentParser(description="Script description")

# Add the root_path argument
parser.add_argument(
    "--model_type",
    type=str,
    default="conditioned",
    help="Model type to use in inference e.g. ('conditioned', 'unconditioned')",
)

parser.add_argument(
    "--model_arquitecture",
    type=str,
    default="tunned",
    help="Model arquitecture to use in inference e.g. ('tunned', 'GAN', 'diffusion)",
)

parser.add_argument(
    "--num_images",
    type=int,
    default=1,
    help="Number of desired images",
)

# Parse the command-line arguments
ARGS = sys.argv

MODEL_TYPE = ARGS[1]
MODEL_ARCHITECTURE = ARGS[2]
NUM_SAMPLES = ARGS[3]
SEED = 52362

# Check the number of parameters
NUM_PARAMETERS = len(sys.argv)
if NUM_PARAMETERS != 0 and NUM_PARAMETERS < 4:
    raise ValueError(
        "Incorrect number of parameters. You should provide either both MODEL_TYPE and MODEL_ARCHITECTURE or none."
    )

# Check if MODEL_ARCHITECTURE is set to "tunned" only for MODEL_TYPE "conditioned"
if MODEL_ARCHITECTURE == "tunned" and MODEL_TYPE != "conditioned":
    print(
        "Warning: Using 'tunned' model architecture with MODEL_TYPE other than 'conditioned'."
    )

# Defining the path were the weights are stored
weights_path = f"/artifacts/{MODEL_TYPE}/{MODEL_ARCHITECTURE}"

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if MODEL_TYPE == "unconditioned":
    if MODEL_ARCHITECTURE == "GAN":
        img_list = []

        # Number of GPUs available.
        if device == "cpu":
            NGPU = 0
        else:
            NGPU = 1

        # Number of channels in the training images. For color images this is 3.
        NC = 3

        # Size of z latent vector (i.e. size of generator input).
        NZ = 100

        # Size of feature maps in generator.
        NGF = 64.0

        # Model initialization
        generator_model = Generator(NGPU, NZ, NC, NGF).to(device)
        generator_model.load_state_dict(torch.load(weights_path, map_location=device))
        generator_model.eval()

        # Create noise to generate the images.
        fixed_noise = torch.randn(NUM_SAMPLES, NZ, 1, 1, device=device)

        # Generate the images from noise.
        fake = generator_model(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Plot images
        plt.imshow(np.transpose(img_list, (1, 2, 0)))
        plt.show()

        # Save the image to a file
        plt.savefig("image.png")

    else:
        # Number of hidden dimension feature
        N_FEAT = 64

        # Context vector is of size
        N_CFEAT = 1

        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        HEIGHT = 64  # 16x16 image

        diffusion_model = ContextUnet(
            in_channels=3, n_feat=N_FEAT, n_cfeat=N_CFEAT, height=HEIGHT
        ).to(device)

        diffusion_model.load_state_dict(torch.load(weights_path, map_location=device))
        diffusion_model.eval()

        # visualize samples (Fast approach)
        plt.clf()
        # samples, intermediate = sample_ddim(32, n=25)
        # animation_ddim = plot_sample(
        #     intermediate, 32, 4, save_dir, "ani_run", None, save=False
        # )

else:
    if MODEL_ARCHITECTURE == "tunned":
        pipe = StableDiffusionPipeline.from_pretrained(
            weights_path, torch_dtype=torch.float16
        ).to(device)
        g_cuda = None
        g_cuda = torch.Generator(device=device)
        g_cuda.manual_seed(SEED)

        prompt = "photo of sks man, anime style"
        negative_prompt = ""
        num_samples = 4
        guidance_scale = 7.5
        num_inference_steps = 50
        height = 512
        width = 512

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

        for img in images:
            display(img)
