# import sys
# sys.path.append("C:/Users/kiraa/simpsons")

import pickle
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytorch_lightning as pl
import tqdm
from omegaconf import DictConfig
from PIL import Image

import wandb
from simpsons.dataset import SimpsonsDataset
from simpsons.model import SimpsonsClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="cfg")
def main(cfg: DictConfig):
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    dir = Path(cfg.inference.dir)
    files = sorted(list(dir.rglob("*.jpg")))
    data = SimpsonsDataset(files=files, mode="test")

    logger = (
        pl.loggers.WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            save_dir=cfg.logging.save_dir,
        ),
    )

    model = SimpsonsClassifier.load_from_checkpoint(cfg.inference.ckpt)
    trainer = pl.Trainer(accelerator="auto", devices="auto")

    for input in tqdm(data):
        output = trainer.predict(model, input.unsqueeze(0).unsqueeze(0))
        predicted_proba = np.max(output) * 100
        y_pred = np.argmax(output)
        predicted_label = label_encoder.classes_[y_pred]
        img = Image.fromarray(
            (
                cv2.normalize(input.permute(1, 2, 0).numpy(), None, 0, 1, norm_type=cv2.NORM_MINMAX)
                * 255
            ).astype(np.uint8)
        )

        logger[0].experiment.log(
            {"image": [wandb.Image(img, caption=f"{str(predicted_label)}+{str(predicted_proba)}")]}
        )


if __name__ == "__main__":
    main()
