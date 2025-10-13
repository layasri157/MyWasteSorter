import os
from pathlib import Path
import shutil
import random
from fastai.vision.all import *
import matplotlib.pyplot as plt

def create_data_folders(dataset_root: Path, project_data_path: Path):
    """
    Detects class folders in dataset_root,
    creates corresponding train, valid, test folders inside project_data_path.
    """
    waste_types = [folder.name for folder in dataset_root.iterdir() if folder.is_dir()]
    print("Detected waste classes:")
    print(waste_types)

    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        for waste_type in waste_types:
            folder = project_data_path / subset / waste_type
            if not folder.exists():
                folder.mkdir(parents=True)

    print(f"Created folder structure in '{project_data_path}'")

    return waste_types

def split_and_move_files(dataset_root: Path, project_data_path: Path, waste_types, train_ratio=0.7, valid_ratio=0.15):
    """
    For each waste class, split its files into train, valid and test folders.
    Moves files accordingly.
    """
    subsets = ['train', 'valid', 'test']

    for waste_type in waste_types:
        source_folder = dataset_root / waste_type
        files = list(source_folder.glob('*.*'))  # all files inside class folder
        random.shuffle(files)
        total_files = len(files)

        train_end = int(total_files * train_ratio)
        valid_end = train_end + int(total_files * valid_ratio)

        train_files = files[:train_end]
        valid_files = files[train_end:valid_end]
        test_files = files[valid_end:]

        # Define destination folders
        dest_folders = {
            'train': project_data_path / 'train' / waste_type,
            'valid': project_data_path / 'valid' / waste_type,
            'test': project_data_path / 'test'  # test folder is flat (all images)
        }

        # Move files
        for f in train_files:
            shutil.copy(f, dest_folders['train'] / f.name)
        for f in valid_files:
            shutil.copy(f, dest_folders['valid'] / f.name)
        for f in test_files:
            shutil.copy(f, dest_folders['test'] / f.name)

    print("Data split and files copied to project folder")

def main():
    # Set your original dataset root folder (with class subfolders)
    dataset_root = Path(r"C:\Users\DELL\Projects\waste_dataset")

    # Your project data folder (where FastAI will load data from)
    project_data_path = Path("data")

    # Create folder structure and get detected classes
    waste_types = create_data_folders(dataset_root, project_data_path)

    # Split and move files based on ratios (optional if not already split)
    # Comment the next line if your data is already split and organized.
    split_and_move_files(dataset_root, project_data_path, waste_types, train_ratio=0.7, valid_ratio=0.15)

    # Debug print counts
    print('Train images:', len(get_image_files(project_data_path/'train')))
    print('Validation images:', len(get_image_files(project_data_path/'valid')))
    print('Test images:', len(get_image_files(project_data_path/'test')))

    # Create DataBlock
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )

    # Create DataLoaders
    dls = data_block.dataloaders(project_data_path, bs=8)

    print('Showing sample batch...')
    dls.show_batch(max_n=9, figsize=(7,7))

    # Create learner and train model
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    print('Training started...')
    learn.fine_tune(10)

    val_loss, val_acc = learn.validate()
    print(f'Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(10,10))

    learn.export('waste_sorter_model.pkl')
    print('Model saved as waste_sorter_model.pkl')

    # Sample prediction on test image
    test_images = list((project_data_path/'test').glob('*.*'))
    if test_images:
        img = PILImage.create(test_images[0])
        pred, pred_idx, probs = learn.predict(img)
        print(f'Example test image predicted as: {pred} with confidence {probs[pred_idx]:.4f}')
    else:
        print('No test images found for prediction.')

if __name__ == '__main__':
    main()
