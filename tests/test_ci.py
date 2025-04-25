def test_imports():
    from simpsons.augmentation import data_augmentation
    from simpsons.classifier import SimpleNet, SimpsonsNet
    from simpsons.dataset import SimpsonsDataset, SimpsonsModule
    from simpsons.model import SimpsonsClassifier

    data_augmentation, SimpleNet, SimpsonsNet, SimpsonsDataset, SimpsonsModule, SimpsonsClassifier
    assert True, "Problems with imports!"
