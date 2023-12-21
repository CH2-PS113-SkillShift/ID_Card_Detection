import os
import random

# Define the path to your dataset
dataset_path = "ocr_dataset"

# Create an empty list to store tuples of (filename, label)
all_entries = []

# Iterate through each folder in the dataset
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Read the labels from the 'labels.txt' file
        labels_file_path = os.path.join(folder_path, "labels.txt")
        with open(labels_file_path, "r") as labels_file:
            lines = labels_file.readlines()

            # Create a list of tuples containing image filenames and labels
            for line in lines:
                words = line.split()
                if len(words) > 1:
                    filename = f"{dataset_path}/{folder_name}/{words[0]}"
                    label = " ".join(words[1:])
                    all_entries.append((filename, label.strip()))


# Define the ratio of entries for training (80%) and validation (20%)
split_ratio = 0.8
split_index = int(len(all_entries) * split_ratio)

# Separate entries into training and validation sets
train_entries = all_entries[:split_index]
val_entries = all_entries[split_index:]


# Function to write entries to a text file
def write_to_file(entries, file_path):
    with open(file_path, "w") as file:
        for filename, label in entries:
            file.write(f"{filename} {label}\n")


# Define paths for annotation files
train_annotation_path = "train_annotation.txt"
val_annotation_path = "val_annotation.txt"

# Write entries to annotation files
write_to_file(train_entries, train_annotation_path)
write_to_file(val_entries, val_annotation_path)

print(f"Training annotation file created: {train_annotation_path}")
print(f"Validation annotation file created: {val_annotation_path}")
