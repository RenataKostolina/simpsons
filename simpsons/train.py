# import sys
# sys.path.append("C:/Users/kiraa/simpsons")

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from simpsons.check_data import ensure_data_downloaded
from simpsons.classifier import SimpsonsNet
from simpsons.dataset import SimpsonsModule
from simpsons.model import SimpsonsClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="cfg")
def main(cfg: DictConfig):
    REQUIRED_PATHS = [cfg.module.test_dir, cfg.module.train_dir]

    if ensure_data_downloaded(REQUIRED_PATHS):
        print("Data ready for processing")
    else:
        print("Failed to download required data")
        exit(1)

    pl.seed_everything(42)
    data_module = SimpsonsModule(
        train_dir=Path(cfg.module.train_dir),
        test_dir=Path(cfg.module.test_dir),
        batch_size=cfg.module.batch_size,
        num_workers=cfg.module.num_workers,
    )
    model = SimpsonsClassifier(SimpsonsNet(n_classes=42), mode="Better")

    loggers = [
        pl.loggers.WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            save_dir=cfg.logging.save_dir,
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=cfg.callbacks.dirpath,
            filename=cfg.callbacks.filename,
            monitor="val_loss",
            save_top_k=1,
            every_n_epochs=1,
        )
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epoch,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
