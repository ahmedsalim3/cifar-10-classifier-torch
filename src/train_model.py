import os
import torch
import argparse

from torch.utils.data import DataLoader

from .dataset import Cifar10Dataset
from .utils import plot_images
from .models import CNNClassifier, VGG16Classifier, train_model
from .utils import (
    plot_training_history,
    visualize_model_results,
    plot_example_predictions,
)
from . import settings


def get_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10 dataset")
    parser.add_argument(
        "--model",
        choices=["vgg16", "cnn"],
        required=True,
        help="Specify which model to train ('vgg16' or 'cnn')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model (default: 30)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    return parser.parse_args()


def train_vgg(train_loader, val_loader, device, class_names, epochs):
    # Initialize the model
    model_class = VGG16Classifier()

    # Train the model and record history
    model, res = train_model(
        model=model_class,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        device=device,
    )

    # Plot training history (loss and accuracy curves)
    fig = plot_training_history(res)
    save_path = os.path.join(settings.RESULTS_PATH, "training_history.png")
    fig.savefig(save_path, dpi=300)

    # Visualize model results (confusion matrix, ROC, precision-recall)
    figures = visualize_model_results(
        model=model, data_loader=val_loader, device=device, class_names=class_names
    )

    save_paths = {
        "cm_fig": os.path.join(settings.RESULTS_PATH, "cm.png"),
        "roc_fig": os.path.join(settings.RESULTS_PATH, "roc_curves.png"),
        "pr_fig": os.path.join(settings.RESULTS_PATH, "pr_curves.png"),
    }

    for fig_key, fig in figures.items():
        if "fig" in fig_key:
            fig.savefig(save_paths[fig_key], dpi=300)

    class_report = figures["class_report"]
    class_report_path = os.path.join(settings.RESULTS_PATH, "class_report.txt")
    with open(class_report_path, "w") as f:
        f.write(class_report)

    # Show some example predictions
    fig2 = plot_example_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        class_names=class_names,
        num_examples=8,
    )
    save_path = os.path.join(settings.RESULTS_PATH, "example_predictions.png")
    fig2.savefig(save_path, dpi=300)

    # Save the trained model
    save_path = os.path.join(settings.MODEL_PATH, "vgg16_classifier.pth")
    torch.save(model.state_dict(), save_path)


def train_cnn(train_loader, val_loader, device, class_names, epochs):
    model_class = CNNClassifier(
        input_shape=(*settings.IMG_SIZE, 3), num_classes=10, learning_rate=1e-3
    )

    model, res = train_model(
        model=model_class,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        device=device,
    )

    # Plot training history (loss and accuracy curves)
    fig = plot_training_history(res)
    save_path = os.path.join(settings.RESULTS_PATH, "training_history.png")
    fig.savefig(save_path, dpi=300)

    # Visualize model results (confusion matrix, ROC, precision-recall)
    figures = visualize_model_results(
        model=model, data_loader=val_loader, device=device, class_names=class_names
    )

    save_paths = {
        "cm_fig": os.path.join(settings.RESULTS_PATH, "cm.png"),
        "roc_fig": os.path.join(settings.RESULTS_PATH, "roc_curves.png"),
        "pr_fig": os.path.join(settings.RESULTS_PATH, "pr_curves.png"),
    }

    for fig_key, fig in figures.items():
        if "fig" in fig_key:
            fig.savefig(save_paths[fig_key], dpi=300)

    class_report = figures["class_report"]
    save_path = os.path.join(settings.RESULTS_PATH, "class_report.txt")
    with open(save_path, "w") as f:
        f.write(class_report)

    # Show some example predictions
    fig2 = plot_example_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        class_names=class_names,
        num_examples=8,
    )
    save_path = os.path.join(settings.RESULTS_PATH, "example_predictions.png")
    fig2.savefig(save_path, dpi=300)

    # Save the trained model
    save_path = os.path.join(settings.MODEL_PATH, "cnn_classifier.pth")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    args = get_args()
    model = args.model
    epochs = args.epochs
    batch_size = args.batch_size

    cifar = Cifar10Dataset()
    train_data, val_data, class_names = cifar.get_datasets()

    fig = plot_images(train_data, num_imgs=4)
    save_path = os.path.join(settings.RESULTS_PATH, "cifar_samples.png")
    fig.savefig(save_path, dpi=300)

    device = settings.DEVICE

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=2
    )

    if model == "vgg16":
        train_vgg(train_loader, val_loader, device, class_names, epochs)
    elif model == "cnn":
        train_cnn(train_loader, val_loader, device, class_names, epochs)
