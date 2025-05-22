from pathlib import Path

import pytest

from simpsons.dataset import SimpsonsModule


@pytest.fixture
def datamodule():
    data_module = SimpsonsModule(
        train_dir=Path("./data/journey-springfield/train"),
        test_dir=Path("./data/journey-springfield/test"),
        batch_size=128,
        num_workers=2,
    )
    datamodule = data_module.prepare_data()
    datamodule = data_module.setup()
    return data_module


@pytest.mark.requires_files
def test_train_dataloaders(datamodule):
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape[0] == 128, "First shape should be batch_size!"
    assert images.shape[1] == 3, "Second shape should be num channels == 3!"
    assert images.shape[2] == 224, "Third shape should be image size!"
    assert len(labels) == 128, "The labels should have the same size as the images!"


@pytest.mark.requires_files
def test_val_dataloaders(datamodule):
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    images, labels = batch
    assert images.shape[0] == 128, "First shape should be batch_size!"
    assert images.shape[1] == 3, "Second shape should be num channels == 3!"
    assert images.shape[2] == 224, "Third shape should be image size!"
    assert len(labels) == 128, "The labels should have the same size as the images!"


@pytest.mark.requires_files
def test_test_dataloaders(datamodule):
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    images = batch
    assert images.shape[0] == 128, "First shape should be batch_size!"
    assert images.shape[1] == 3, "Second shape should be num channels == 3!"
    assert images.shape[2] == 224, "Third shape should be image size!"
