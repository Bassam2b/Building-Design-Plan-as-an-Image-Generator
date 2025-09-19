import pandas as pd
from PIL import Image
import os

#  Configuration 
dataset_path = 'dataset'
image_folder = os.path.join(dataset_path, 'images')
csv_path = os.path.join(dataset_path, 'metadata.csv')

# Verification Script 
def verify_dataset():
    print(" Starting Dataset Verification ")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: metadata.csv not found at '{csv_path}'")
        return
    print(" metadata.csv found.")
    
    # Load the CSV and check its structure
    try:
        df = pd.read_csv(csv_path)
        if 'file_name' not in df.columns or 'text' not in df.columns:
            print("Error: CSV must contain 'file_name' and 'text' columns.")
            return
        print(" CSV columns are correct.")
    except Exception as e:
        print(f"Error reading or parsing CSV file: {e}")
        return

    # Check each image listed in the CSV
    all_images_valid = True
    for index, row in df.iterrows():
        image_name = row['file_name']
        image_path = os.path.join(image_folder, image_name)
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f" Error: Image '{image_name}' listed in CSV but not found in images folder.")
            all_images_valid = False
            continue
            
        # Check image dimensions
        try:
            with Image.open(image_path) as img:
                if img.size != (512, 512):
                    print(f" Warning: Image '{image_name}' is not 512x512 pixels. It is {img.size}.")
                    all_images_valid = False
        except Exception as e:
            print(f" Error: Could not open or read image '{image_name}'. Error: {e}")
            all_images_valid = False

    if all_images_valid:
        print("\n --- Verification Complete: All checks passed! --- ")
        print(f"Successfully verified {len(df)} entries.")
    else:
        print("\n Verification Complete: Issues found. Please fix the errors listed above. ")

if __name__ == "__main__":
    verify_dataset()