from PIL import Image
import os

# Configuration 
# Folder with your original, various-sized images
input_folder = "dataset/raw_images" 

# Folder where the processed 512x512 images will be saved
output_folder = "dataset/images"

# The target size for the model
target_size = (512, 512)

# Preprocessing Script
def preprocess_images():
    """
    Crops and resizes all images from the input folder to the target size
    and saves them in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    image_files = os.listdir(input_folder)
    print(f"Found {len(image_files)} images to process.")

    for filename in image_files:
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with Image.open(input_path) as img:
                # Convert to RGB to handle various image formats like PNG with alpha channels
                img = img.convert("RGB")

                # Crop to a square from the center
                width, height = img.size
                short_side = min(width, height)
                
                left = (width - short_side) / 2
                top = (height - short_side) / 2
                right = (width + short_side) / 2
                bottom = (height + short_side) / 2
                
                img_cropped = img.crop((left, top, right, bottom))
                
                # Resize the square to the target size
                img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save the final image
                img_resized.save(output_path)
                print(f"Processed and saved: {filename}")

        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")

    print("\n Preprocessing Complete ")

if __name__ == "__main__":
    preprocess_images()