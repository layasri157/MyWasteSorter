import splitfolders
import os

# YOUR CURRENT paths (MyWasteSorter folder)
input_folder = r"C:\Users\DELL\MyWasteSorter\waste_dataset"      # Extracted ZIP
output_folder = r"C:\Users\DELL\MyWasteSorter\data"             # train/val/test

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Split 70/15/15
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

print(f"✅ Split complete!")
print(f"📁 Train: {len(os.listdir(os.path.join(output_folder, 'train')))} classes")
print(f"📁 Total folders: data/train/, data/val/, data/test/")
