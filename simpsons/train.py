from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from simpsons.dataset import init_dataloader
from simpsons.model import SimpsonsClassifier


def fit_epoch(model, train_loader, criterion, optimizer, scheduler):
    """
    Model training for one epoch

    Args:
        model
        train_loader
        criterion
        optimizer
        scheduler

    Returns:
        Train loss, train accuracy

    """
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    scheduler.step()
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    """
    Validation of the model for one epoch

    Args:
        model
        val_loader
        criterion

    Returns:
        Validation loss, validation accuracy

    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc


# определим директории с тренировочными и тестовыми файлами
TRAIN_DIR = Path("./train/")
TEST_DIR = Path("./test/")


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler):
    """
    Model training

    """
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, scheduler)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(
                log_template.format(
                    ep=epoch + 1,
                    t_loss=train_loss,
                    v_loss=val_loss,
                    t_acc=train_acc,
                    v_acc=val_acc,
                )
            )

    return history


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = SimpsonsClassifier()
    batch_size = 256
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    train_loader, val_loader = init_dataloader(TRAIN_DIR, batch_size=batch_size)
    history = train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler)

    return history


if __name__ == "__main__":
    main()
