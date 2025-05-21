import hydra
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
from omegaconf import DictConfig

from simpsons.model import SimpsonsClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="cfg")
def convert_to_onnx(cfg: DictConfig):
    model = SimpsonsClassifier.load_from_checkpoint(cfg.inference.ckpt)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["PREPROCESSED_IMAGE"],
        output_names=["OUTPUT"],
        dynamic_axes={"PREPROCESSED_IMAGE": {0: "batch_size"}, "OUTPUT": {0: "batch_size"}},
    )

    print("Model successfully converted to ONNX")


def check_onnx(onnx_model_path):
    inputs = torch.randn(1, 3, 224, 224)

    ort_sess = ort.InferenceSession("model.onnx")
    outputs = ort_sess.run(None, {"PREPROCESSED_IMAGE": inputs.numpy().astype(np.float32)})
    print("ONNX model check passed!")


if __name__ == "__main__":
    convert_to_onnx()
    check_onnx("model.onnx")
