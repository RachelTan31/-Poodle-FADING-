import os
import argparse

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

from FADING_util import util   # optional now, but harmless
from p2p import *
from null_inversion import *

# -------------------------------------------------
# Argparse
# -------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--image_path', required=True, help='Path to input image')
parser.add_argument('--age_init', required=True, type=int,
                    help='Specify the initial age in MONTHS')
parser.add_argument('--specialized_path', required=True,
                    help='Path to specialized diffusion model (poodle_fading_model)')
parser.add_argument('--save_aged_dir', default='./outputs',
                    help='Path to save outputs')
parser.add_argument('--target_ages', nargs='+', type=int,
                    default=[6, 36, 84],
                    help='Target ages in MONTHS')

args = parser.parse_args()

# -------------------------------------------------
# Parse args
# -------------------------------------------------
image_path      = args.image_path
age_init        = args.age_init          # months
specialized_path = args.specialized_path
save_aged_dir   = args.save_aged_dir
target_ages     = args.target_ages       # list of months

if not os.path.exists(save_aged_dir):
    os.makedirs(save_aged_dir)

# We don't use gender anymore; just poodles 🙂
poodle_placeholder = "poodle"
inversion_prompt = f"photo of {age_init} month old {poodle_placeholder}"

input_img_name = os.path.splitext(os.path.basename(image_path))[0]

# -------------------------------------------------
# Load specialized diffusion model
# -------------------------------------------------
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
g_cuda = torch.Generator(device=device)

ldm_stable = StableDiffusionPipeline.from_pretrained(
    specialized_path,
    scheduler=scheduler,
    safety_checker=None
).to(device)
tokenizer = ldm_stable.tokenizer

# -------------------------------------------------
# Null-text inversion on the input image
# -------------------------------------------------
null_inversion = NullInversion(ldm_stable)
(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(
    image_path,
    inversion_prompt,
    offsets=(0, 0, 0, 0),
    verbose=True
)

# -------------------------------------------------
# Age editing loop
# -------------------------------------------------
for age_new in target_ages:
    print(f'Age editing with target age {age_new} months...')

    new_prompt = f"photo of {age_new} month old poodle"

    # Prompt-to-Prompt blend configuration
    blend_word = (((str(age_init), "poodle",), (str(age_new), "poodle",)))
    is_replace_controller = True
    prompts = [inversion_prompt, new_prompt]

    cross_replace_steps = {'default_': 0.8}
    self_replace_steps = 0.5

    eq_params = {"words": (str(age_new),), "values": (1,)}

    controller = make_controller(
        prompts,
        is_replace_controller,
        cross_replace_steps,
        self_replace_steps,
        tokenizer,
        blend_word,
        eq_params
    )

    images, _ = p2p_text2image(
        ldm_stable,
        prompts,
        controller,
        generator=g_cuda.manual_seed(0),
        latent=x_t,
        uncond_embeddings=uncond_embeddings
    )

    new_img = images[-1]
    new_img_pil = Image.fromarray(new_img)
    out_path = os.path.join(save_aged_dir, f'{input_img_name}_{age_new}m.png')
    new_img_pil.save(out_path)
    print("Saved:", out_path)
