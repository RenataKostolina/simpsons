import pickle

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from simpsons.augmentation import data_augmentation

# разные режимы датасета
DATA_MODES = ["train", "val", "test"]
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224


class SimpsonsDataset(Dataset):
    """
    A dataset with images that simultaneously downloads them from folders,
    scales them and turns them into torch tensors.

    """

    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != "test":
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open("label_encoder.pkl", "wb") as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype="float32")
        x = transform(x)
        if self.mode == "test":
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


def init_dataloader(TRAIN_DIR: str, batch_size: int, num_workers: int = 6):
    """
    Initializing dataloaders for training and validation samples

    Args:
        TRAIN_DIR: The directory where the files are located
        batch_size (int): -
        num_workers (int, optional): -

    Returns:
        train_loader: torch.utils.data.Dataloader
        val_loader: torch.utils.data.Dataloader
    """
    train_val_files = sorted(list(TRAIN_DIR.rglob("*.jpg")))
    train_val_labels = [path.parent.name for path in train_val_files]

    train_val_files, train_val_labels = data_augmentation(train_val_labels)
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )

    train_dataset = SimpsonsDataset(train_files, mode="train")
    val_dataset = SimpsonsDataset(val_files, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def init_test_loader(TEST_DIR: str, batch_size: int, num_workers: int = 6):
    """
    Initializing dataloaders for a test sample
    Args:
        TEST_DIR: The directory where the files are located
        batch_size (int): -
        num_workers (int, optional): -

    Returns:
        test_loader: torch.utils.data.Dataloader
        val_loader: torch.utils.data.Dataloader
    """
    test_files = sorted(list(TEST_DIR.rglob("*.jpg")))
    test_dataset = SimpsonsDataset(test_files, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader
