### Image Processing Using Python

# PIL -> Python Image Library
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

pic = Image.open('white-flower.jpg')
# pic

pic_arr = np.asarray(pic)
# pic_arr
pic_arr.shape

plt.imshow(pic_arr)
# plt.axis(False);

pic_red = pic_arr.copy()
pic_green = pic_arr.copy()
pic_blue = pic_arr.copy()

# RGB
pic_red[:,:,1] = 0 # GREEN Channel : set -> 0
pic_red[:,:,2] = 0 # BLUE Channel : set -> 0

# RGB
pic_green[:,:,0] = 0 # RED Channel : set -> 0
pic_green[:,:,2] = 0 # BLUE Channel : set -> 0

# RGB
pic_blue[:,:,0] = 0 # RED Channel : set -> 0
pic_blue[:,:,1] = 0 # GREEN Channel : set -> 0

# pic_red
# pic_green
pic_blue

import cv2 # cv2 is opencv plugin for python

img = cv2.imread('/content/white-flower.jpg')
img

img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img_rgb

img_gray = cv2.imread('/content/white-flower.jpg',cv2.IMREAD_GRAYSCALE)
# img_gray

plt.imshow(img_gray,cmap="gray")

img_resize = cv2.resize(img , (200,400))
plt.imshow(img_resize)

w_ratio = 0.5
h_ratio = 0.5

new_img = cv2.resize(img , (0,0),fx=w_ratio,fy=h_ratio)
# new_img  # Image resized
plt.imshow(new_img)

"""### Generating Faces using Diffusion Models"""

from diffusers import DDPMPipeline # Denoising Diffusion Probablistic Model

ddpm = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda")

image = ddpm(num_inference_steps=30).images[0]

image

from diffusers import DDPMScheduler , UNet2DModel

scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
# scheduler

model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to("cuda")

scheduler.set_timesteps(50)

sample_size = model.config.sample_size
sample_size

noise = torch.randn((1,3,sample_size,sample_size) , device="cuda")
input = noise

for t in scheduler.timesteps:
  with torch.inference_mode():  # enable and trun on inference mode (turning off backtracking)
      noisy_residual = model(input,t).sample

      previous_noisy_sample = scheduler.step(noisy_residual,t,input).prev_sample

      input = previous_noisy_sample

image = (input / 2 + 0.5).clamp(0,1).squeeze()
image = (image.permute(1,2,0) * 255 ).round().to(torch.uint8).cpu().numpy()

image = Image.fromarray(image)
image

"""### Generation images using a prompt with Diffusion Models"""

import torch
from diffusers import DiffusionPipeline

torch.cuda.empty_cache()

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
    # device_map="cuda"
    ).to("cuda")

pipe.enable_model_cpu_offload()   # Important fix
pipe.enable_attention_slicing()   # reduces VRAM

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt,height=512 , width=512).images[0]
image = pipe(
    prompt,
    height=768,
    width=768,
    num_inference_steps=20
).images[0]

image