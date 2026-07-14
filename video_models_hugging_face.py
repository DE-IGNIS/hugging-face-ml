### Video Models

import torch
import diffusers
import transformers
import cv2

"""### Generating Video With Diffusion"""

from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)

path = "elmo.jpg"

from diffusers.utils import load_image, export_to_video

image = load_image(path)

pipe.enable_model_cpu_offload()  # to remove out of memory error

generator = torch.manual_seed(42)

frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

# len(frames)
export_to_video(frames, "elmo.mp4", fps=7)

"""### Generating Video using another diffusion model (Using Image & Prompt) , ali-vilab is no longer maintained and hence is irrelevant with newer diffuser models"""

torch.cuda.empty_cache()
from diffusers import I2VGenXLPipeline

repo_id = "ali-vilab/i2vgen-xl"

pipeline = I2VGenXLPipeline.from_pretrained(
    repo_id, torch_dtype=torch.float16, variant="fp16"
)

pipeline.enable_model_cpu_offload()

prompt = "The sea waves heavily"
generator = torch.manual_seed(42)

frames = pipeline(prompt=prompt, image=sea, num_frames=16, generator=generator).frames[
    0
]

# import diffusers, transformers
# print(diffusers.__version__)
# print(transformers.__version__)

"""### Generating Video using another diffusion model (Using Image & Prompt)"""

from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import load_image

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")

pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16
).to("cuda")

sea = load_image("sea.jpg")
# sea

generator = torch.manual_seed(42)

frames = pipe(
    prompt="stormy ocean with dark clouds, cinematic lighting",
    image=sea,  # your input image
    num_frames=16,
    guidance_scale=7.5,
    generator=generator,
).frames[0]

# len(frames)
export_to_video(frames, "sea.mp4", fps=7)
