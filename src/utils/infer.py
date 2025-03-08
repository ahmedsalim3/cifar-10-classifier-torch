import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from PIL import Image

from .. import settings

device = settings.DEVICE


def predict_and_visualize(model, image_path, top_N=1, requests=False):
    """
    Make predictions on an image and visualize the results
    """

    model.eval()

    transform = Compose(
        [
            Resize(settings.IMG_SIZE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load image
    if requests:
        try:
            import requests
            from io import BytesIO

            response = requests.get(image_path)
            response.raise_for_status()
            org_img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
    else:
        try:
            org_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image from file: {e}")
            return None

    img = transform(org_img).unsqueeze(0)
    img = img.to(device)

    try:
        # Make prediction
        with torch.no_grad():
            predictions = model(img)

            # Handle the case where model output is already normalized (softmax applied)
            if predictions.sum().item() == 1.0:
                probabilities = predictions[0]
            else:
                probabilities = torch.nn.functional.softmax(predictions, dim=1)[0]

        top_indices = torch.argsort(probabilities, descending=True)[:top_N]
        top_labels = [
            settings.label_dict.get(idx.item(), f"Class {idx.item()}") for idx in top_indices
        ]
        top_confidences = [probabilities[idx].item() for idx in top_indices]

        # Visualize results
        plt.figure(figsize=(10, 6))
        plt.imshow(org_img)
        plt.axis("off")
        plt.title(
            "\n".join(
                [
                    f"{label}: {conf:.2f}"
                    for label, conf in zip(top_labels, top_confidences)
                ]
            ),
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()

        return {"top_labels": top_labels, "top_confidences": top_confidences}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def batch_predict(model, image_paths):
    """
    Make predictions on multiple images
    """

    # Ensure model is in eval mode
    model.eval()

    # Define image transformation
    transform = Compose(
        [
            Resize(settings.IMG_SIZE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    results = []

    for path in image_paths:
        try:

            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO

                response = requests.get(path)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(path).convert("RGB")

            img_tensor = transform(img).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                predictions = model(img_tensor)
                if predictions.sum().item() == 1.0:
                    probabilities = predictions[0]
                else:
                    probabilities = torch.nn.functional.softmax(predictions, dim=1)[0]

            indices = torch.argsort(probabilities, descending=True)
            labels = [
                settings.label_dict.get(idx.item(), f"Class {idx.item()}") for idx in indices
            ]
            confidences = [probabilities[idx].item() for idx in indices]

            results.append(
                {
                    "path": path,
                    "predicted_class": labels[0],
                    "confidence": confidences[0],
                    "all_predictions": [
                        {"label": l, "confidence": c}
                        for l, c in zip(labels, confidences)
                    ],
                }
            )

        except Exception as e:
            print(f"Error processing {path}: {e}")
            results.append({"path": path, "error": str(e)})

    return results
