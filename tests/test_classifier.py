import torch

from simpsons.classifier import SimpleNet, SimpsonsNet


def test_model_prediction():
    model = SimpsonsNet(n_classes=42)
    dummy_input = torch.rand(1, 3, 224, 224)

    prediction = model(dummy_input)
    assert prediction.shape == (1, 42), "Output should be (batch_size, num_classes)!"


def test_base_prediction():
    model = SimpleNet(n_classes=42)
    dummy_input = torch.rand(1, 3, 224, 224)

    prediction = model(dummy_input)
    assert prediction.shape == (1, 42), "Output should be (batch_size, num_classes)!"
