import torch
from diffusers import DiffusionPipeline

# The folder where your fine-tuned LoRA weights are saved
lora_model_path = "lora-floorplan-model" 

# Check if a CUDA-enabled GPU is available and set the device
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")

# Load the base pre-trained diffusion pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",  
    torch_dtype=torch.float16 
)

# Load your fine-tuned LoRA weights into the UNet model of the pipeline.
print(f"Loading LoRA weights from: {lora_model_path}")
pipe.load_lora_weights(lora_model_path)

#  Move the entire pipeline (with LoRA) to the GPU
pipe.to(device)

# Define the prompt for image generation
prompt = "2D isometric floor plan of a 85m² Scandinavian minimalist apartment. Entrance on North leads to open living/kitchen area. Two bedrooms placed on South and West walls. Master bedroom (SW) has ensuite bathroom. Kitchen on South wall. Annotated dimensions. Compass rose. Clean, airy aesthetic with light wood tones. Square composition."

# Generate the image
print("Generating image with LoRA...")
image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# Save the generated image
output_filename = "creativeAgencyOffice_Generated_Floor_Plan_with_LoRA.png"
image.save(output_filename)

print(f"Image saved as {output_filename}")
