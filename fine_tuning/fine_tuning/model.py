import torch
from diffusers import DiffusionPipeline
from torch import autocast

# Check device 
if torch.cuda.is_available():
    device = "cuda"
    print(f" Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(" CUDA not available, using CPU instead")

# Load pre-trained Stable Diffusion 
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16"
)

# Optimize memory for low-VRAM GPUs 
pipe.enable_attention_slicing()

# Move pipeline to device (GPU or CPU) 
pipe.to(device)

# Define your prompt 
prompt = "a photo of an astronaut riding a horse on mars"

# Generate image
with autocast(device):
    image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# Save output 
image.save("astronaut_horse.png")
print(" Image saved as astronaut_horse.png")
