import pickle

import numpy as np
import pytorch_lightning as pl
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

    Args:
        files: List of files to download
        mode: Operating mode, may have a value of "train", "val" or "test"
    """

    def __init__(self, files, mode):
        super().__init__()
        self.files = sorted(files)
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


class SimpsonsModule(pl.LightningDataModule):
    """
    A DataModule standardizes training, validation, testing, data preparation,
    and transforms. The main advantage of this is consistent data splitting,
    data preparation, and transformations across models.

    Args:
        train_dir (str): The directory where the train and validation files are located
        test_dir (str): The directory where the test files are located
        batch_size (int): -
        num_workers (int, optional): -
    """

    def __init__(self, train_dir: str, test_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.TRAIN_DIR = train_dir
        self.TEST_DIR = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Use this to download and prepare data. Lightning ensures this method
        is called only within a single process, so you can safely add your
        downloading logic within.

        Return: None
        Result: Save names of train and validation files in self.train_val_files,
        save names of test files in self.test_files

        """
        self.train_val_files = sorted(list(self.TRAIN_DIR.rglob("*.jpg")))
        self.test_files = sorted(list(self.TEST_DIR.rglob("*.jpg")))

    def setup(self, stage=None):
        """
        This function is used to add augmentation for training and validation
        files, and then create datasets.

        Return: None
        Result: Save datasets for training, validation and test in Class.
        """
        train_val_labels = [path.parent.name for path in self.train_val_files]

        train_val_files, train_val_labels = data_augmentation(
            train_val_labels, self.train_val_files, self.TRAIN_DIR
        )
        train_files, val_files = train_test_split(
            train_val_files, test_size=0.25, stratify=train_val_labels
        )

        self.train_dataset = SimpsonsDataset(train_files, mode="train")
        self.val_dataset = SimpsonsDataset(val_files, mode="val")
        self.test_dataset = SimpsonsDataset(self.test_files, mode="test")

    def train_dataloader(self):
        """
        Initializing dataloaders for training samples from self.train_dataset
        with parameters of Class, such as batch_size and num_workers

        Returns:
            train_loader: torch.utils.data.Dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Initializing dataloaders for validation samples from self.val_dataset
        with parameters of Class, such as batch_size and num_workers

        Returns:
            val_loader: torch.utils.data.Dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Initializing dataloaders for test samples from self.test_dataset
        with parameters of Class, such as batch_size and num_workers

        Returns:
            test_loader: torch.utils.data.Dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
