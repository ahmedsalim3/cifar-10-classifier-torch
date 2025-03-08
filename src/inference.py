import argparse
import logging
import torch

from .models import VGG16Classifier, CNNClassifier
from . import settings
from .utils import predict_and_visualize, batch_predict

if __name__ == "__main__":
    device = settings.DEVICE

    parser = argparse.ArgumentParser(
        description="Run image inference with CNN or VGG model"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "vgg"],
        required=True,
        help="Choose the model: cnn or vgg",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image (URL or local file)",
    )
    parser.add_argument(
        "--requests",
        action="store_true",
        help="If the image is a URL, use requests to fetch it",
    )

    args = parser.parse_args()

    if args.model == "vgg":
        model = VGG16Classifier()
        model_path = settings.MODEL_PATH / "vgg16_classifier.pth"
    else:
        model = CNNClassifier()
        model_path = settings.MODEL_PATH / "cnn_classifier.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    logging.info("Model loaded successfully.")

    result = predict_and_visualize(
        model=model,
        image_path=args.image_path,
        requests=args.requests,
    )
    logging.info(result)
