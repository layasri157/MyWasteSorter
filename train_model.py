from fastai.vision.all import *
from pathlib import Path

# Use only the clean class names (avoid duplicates like "1-Cardboard")
VALID_CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

def is_clean_class_folder(p: Path):
    return p.name in VALID_CLASSES

def get_items_fn(path: Path):
    # Get images only from clean class folders under train/val
    return get_image_files(path).filter(lambda p: is_clean_class_folder(p.parent))

def main():
    path = Path("data")

    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_items_fn,
        splitter=GrandparentSplitter(train_name="train", valid_name="val"),
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)],
    )

    dls = data_block.dataloaders(path, bs=16)

    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(10)

    val_loss, val_acc = learn.validate()
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    learn.export("waste_sorter_model.pkl")
    print(f"Model saved at: {learn.path / 'waste_sorter_model.pkl'}")

if __name__ == "__main__":
    main()
