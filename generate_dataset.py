# %%
import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont


# %%
# Function to generate random text (single character)
def generate_random_text(characters, used_characters):
    remaining_characters = list(set(characters) - set(used_characters))
    if not remaining_characters:
        # Reset used_characters if all characters have been used
        used_characters = []
        remaining_characters = characters
    return random.choice(remaining_characters), used_characters


# Function to generate random image
def generate_random_image(
    font_path, text, font_size, image_size=(40, 40), noise_level=30
):
    # Create a blank image
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Load font
    font = ImageFont.truetype(font_path, font_size)

    draw.text((7, 2), text, font=font, fill="black", spacing=2)

    # Convert to NumPy array
    img_array = np.array(image)

    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    img_array = np.clip(
        img_array.astype(np.float32)
        + np.random.normal(0, noise_level, img_array.shape),
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(img_array)


# Generate dataset
dataset_size = 50000
output_folder = "generated_dataset"
font_path1 = "arial.ttf"
font_path2 = "OCRAEXT.TTF"

# Characters string
characters = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

# Delete existing folder if it exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Create a new output folder
os.makedirs(output_folder)

# Create CSV file for annotations
csv_data = {"Image_Name": [], "Label": []}

# Keep track of used characters
used_characters = []

print("Generating data...")

for i in range(dataset_size):
    text, used_characters = generate_random_text(characters, used_characters)
    font_size = 30
    font_path = font_path1 if i % 2 == 0 else font_path2  # Alternate between fonts

    image = generate_random_image(font_path, text, font_size)
    image_name = f"{i:06d}.jpg"
    image_path = os.path.join(output_folder, image_name)

    # Save image
    image.save(image_path)

    # Update CSV data
    csv_data["Image_Name"].append(image_name)
    csv_data["Label"].append(text)

# Create DataFrame and save CSV
df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(output_folder, "annotations.csv"), index=False)
print("Finished")


# %%


# %%
