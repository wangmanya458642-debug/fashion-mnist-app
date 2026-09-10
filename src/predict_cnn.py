from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .model_cnn import FashionCNN


DEVICE = torch.device("cpu")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_fashion_cnn.pth"


@lru_cache(maxsize=1)
def load_model(checkpoint_path=MODEL_PATH):
    """Load the CNN checkpoint once for CPU inference."""
    model = FashionCNN()
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_array):
    """Convert a 28x28 grayscale image with values in [0, 255] to a tensor."""
    image_array = np.asarray(image_array)
    if image_array.shape != (28, 28):
        raise ValueError(f"Expected a 28x28 grayscale image, got {image_array.shape}.")
    if image_array.min() < 0 or image_array.max() > 255:
        raise ValueError("Image values must be in the range [0, 255].")

    normalized = image_array.astype(np.float32) / 255.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def predict_cnn(image_array):
    """Return the predicted Fashion-MNIST class index."""
    tensor = preprocess_image(image_array)
    with torch.no_grad():
        logits = load_model()(tensor)
    return int(torch.argmax(logits, dim=1).item())
