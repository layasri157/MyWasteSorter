import splitfolders  # make sure split-folders package is installed

input_folder = r"C:\Users\DELL\Projects\waste_dataset"  # folder where images are extracted
output_folder = r"C:\Users\DELL\Projects\MyWasteSorter\data"  # destination for train/val/test folders

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

