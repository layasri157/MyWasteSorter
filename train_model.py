from fastai.vision.all import *
from pathlib import Path

def main():
    # Path to dataset folder
    path = Path('data')

    # Define data block with train/valid split and item transforms
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )

    # Create DataLoaders
    dls = data_block.dataloaders(path, bs=16)

    # Show sample batch (optional)
    dls.show_batch(max_n=9, figsize=(7,7))

    # Create Learner with resnet34 and accuracy metric
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    # Train model with fine_tune for 10 epochs
    learn.fine_tune(10)

    # Validate and print loss and accuracy
    val_loss, val_acc = learn.validate()
    print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

    # Export/save the trained model
    learn.export('waste_sorter_model.pkl')
    print(f'Model saved at: {learn.path / "waste_sorter_model.pkl"}')

if __name__ == '__main__':
    main()
