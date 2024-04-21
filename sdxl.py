import argparse
from diffusers import DiffusionPipeline
import torch

def main(prompt):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    images = pipe(prompt=prompt).images[0]
    images.save('image.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--prompt", type=str, default="Compression artifacts",
                        help="Prompt for generating the image.")
    args = parser.parse_args()
    main(args.prompt)
