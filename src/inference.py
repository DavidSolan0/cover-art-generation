from inference_utilities import generate_images


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using StableDiffusionPipeline"
    )
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for image generation",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of image samples to generate"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for image generation",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of generated images"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of generated images"
    )
    args = parser.parse_args()

    images = generate_images(
        args.prompt,
        args.negative_prompt,
        args.num_samples,
        args.guidance_scale,
        args.num_inference_steps,
        args.height,
        args.width,
    )

    for img in images:
        display(img)


if __name__ == "__main__":
    main()
