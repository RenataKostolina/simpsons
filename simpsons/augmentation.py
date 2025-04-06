import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_image(file: str):
    """
    Uploads the image and returns it

    Args:
        file (str): File name
    """
    image = Image.open(file)
    image.load()
    return image


augmenters = transforms.RandomChoice(
    [
        transforms.Compose([transforms.Resize(size=224), transforms.RandomCrop(180)]),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomPerspective(distortion_scale=0.7),
    ]
)


def data_augmentation(train_labels, train_files, TRAIN_DIR):
    """
    Adds augmentation to data using RandomRotation, RandomHorizontalFlip,
    and RandomPerspective transformations

    Args:
        train_labels: Class labels for each object from the training set
        train_files: Names of files with images for training
        TRAIN_DIR: The directory where the training files are located

    Returns:
        Names of augmented files, class labels for augmented files

    """
    data = pd.DataFrame(train_labels, columns=["name"])
    data["count"] = 1
    data = data.groupby("name").count().sort_values("count")

    data["iteration"] = 1500 - data["count"]
    data.loc[data["iteration"] < 0, "iteration"] = 0

    for image_path in tqdm(train_files):
        path = image_path.parents[0]
        character = image_path.parent.name
        img = load_image(image_path)
        num_iter = round(data.loc[character]["iteration"] / data.loc[character]["count"])

        for i in range(num_iter):
            aug_img = augmenters(img)
            aug_img.save(
                f"{path}/{image_path.name.split('.')[0]}"
                + "_"
                + (4 - len(str(i + 1))) * "0"
                + f"{i+1}.jpg"
            )

    aug_train_files = sorted(list(TRAIN_DIR.rglob("*.jpg")))
    aug_train_labels = [path.parent.name for path in aug_train_files]
    return aug_train_files, aug_train_labels
