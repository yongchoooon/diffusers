import torch
import os
import json
import random
from diffusers import DiffusionPipeline
from PIL import Image 
from tqdm import tqdm

sd_pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision=None,
)

model_name = "sd14-lora-fire-aihub-new-chatgpt"
prompt_json_path = os.path.join(model_name, 'BLIP+chatgpt_prompt_real_new_train_lora_cleaned.json')

with open(prompt_json_path, 'r') as f:
    prompt_json_data = json.load(f)
    
sd_pipe.unet.load_attn_procs(model_name)

os.makedirs(os.path.join('inference', model_name), exist_ok = True)

device = torch.device("cuda:0")
sd_pipe.to(device)

images = []

for item in tqdm(prompt_json_data):
    path = item['path']
    prompt = item['best_n'][:-1]
    seed = random.randint(1000000, 9999999)
    generator = torch.Generator(device=device).manual_seed(seed)

    img = sd_pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    img_name = path.split('/')[-1].split('.')[0] + '.jpg'

    img.save(os.path.join('inference', model_name, img_name), 'jpeg')