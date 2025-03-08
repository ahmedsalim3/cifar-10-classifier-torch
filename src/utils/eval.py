import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
import seaborn as sns
from itertools import cycle

from .. import settings


def get_predictions(model, data_loader, device):
    """
    Obtain model predictions, true labels, and probability scores from a given data loader

    Args:
        model: Trained model used for inference
        data_loader: DataLoader providing input samples and corresponding labels

    Returns:
        Tuple containing:
        - all_predictions (numpy array): Predicted class labels
        - all_targets (numpy array): True class labels
        - all_probs (numpy array): Model output probabilities
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = outputs.detach().cpu().numpy()
            _, predicted = torch.max(outputs, 1)

            all_probs.append(probs)
            all_predictions.append(predicted.cpu().numpy())
            if len(targets.shape) == 1:  # If not one-hot encoded
                all_targets.append(targets.cpu().numpy())
            else:  # If one-hot encoded
                _, targets_idx = torch.max(targets, 1)
                all_targets.append(targets_idx.cpu().numpy())

    return (
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


def plot_training_history(results):
    """
    Plot the training and validation loss and accuracy over multiple epochs

    Args:
        results: An instance of TrainingResults containing recorded metrics
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # losses
    ax1.plot(results.epochs, results.train_losses, label="Training Loss")
    if results.val_losses:
        ax1.plot(results.epochs, results.val_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # accuracies
    ax2.plot(results.epochs, results.train_accuracies, label="Training Accuracy")
    if results.val_accuracies:
        ax2.plot(results.epochs, results.val_accuracies, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(settings.RESULTS_PATH, "training_history.png")
    plt.savefig(save_path)
    plt.show()


def visualize_model_results(model, data_loader, device, class_names=None):
    """
    Generate various evaluation plots and metrics for a trained model.

    Args:
        model: Trained model for evaluation.
        data_loader: DataLoader providing evaluation data.
        class_names (optional): List of class names for labeling outputs.

    Returns:
        Dictionary containing predictions, true labels, and probability scores.
    """

    y_pred, y_true, y_probs = get_predictions(model, data_loader, device)

    __confusion_matrix(y_true, y_pred, class_names)
    __roc_curve(y_true, y_probs, class_names)
    __pr_curve(y_true, y_probs, class_names)
    __class_report(y_true, y_pred, class_names)

    return {"y_pred": y_pred, "y_true": y_true, "y_probs": y_probs}


def plot_example_predictions(
    model, data_loader, device, class_names=None, num_examples=5
):
    """
    Display example model predictions alongside true labels.

    Args:
        model: Trained model used for inference.
        data_loader: DataLoader providing input samples and labels.
        class_names (optional): List of class names for labeling the images.
        num_examples (int, optional): Number of example predictions to display (default: 5).
    """
    model.eval()
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    predicted = predicted.cpu().numpy()

    if len(labels.shape) > 1:
        _, labels = torch.max(labels, 1)

    if class_names is None:
        class_names = [str(i) for i in range(outputs.shape[1])]

    images_per_row = 5
    num_examples = min(num_examples, len(images))
    rows = (num_examples + images_per_row - 1) // images_per_row

    fig = plt.figure(figsize=(images_per_row * 3, rows * 3))

    save_path = os.path.join(settings.RESULTS_PATH, "example_predictions.png")

    for i in range(num_examples):

        ax = fig.add_subplot(rows, images_per_row, i + 1)

        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        is_correct = predicted[i] == labels[i].item()
        title_color = "green" if is_correct else "red"

        ax.set_title(
            f"Pred: {class_names[predicted[i]]}\nTrue: {class_names[labels[i]]}",
            color=title_color,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def __confusion_matrix(y_true, y_pred, class_names=None):
    """
    Generate and display a confusion matrix for model predictions.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        class_names (optional): List of class names for labeling the matrix.

    Returns:
        Confusion matrix as a numpy array.
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    # normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    save_path = os.path.join(settings.RESULTS_PATH, "cm.png")

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return cm


def __roc_curve(y_true, y_probs, class_names=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for multi-class classification.

    Args:
        y_true: True class labels.
        y_probs: Model output probability scores.
        class_names (optional): List of class names for labeling the curves.

    Returns:
        Dictionary containing Area Under Curve (AUC) values for each class.
    """
    n_classes = y_probs.shape[1]

    y_true_bin = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(
        [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )
    save_path = os.path.join(settings.RESULTS_PATH, "roc_curves.png")

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve (class {class_names[i]}) (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return roc_auc


def __pr_curve(y_true, y_probs, class_names=None):
    """
    Plot Precision-Recall curves for multi-class classification.

    Args:
        y_true: True class labels.
        y_probs: Model output probability scores.
        class_names (optional): List of class names for labeling the curves.

    Returns:
        Tuple containing dictionaries of precision and recall values for each class.
    """
    n_classes = y_probs.shape[1]

    y_true_bin = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(
            y_true_bin[:, i], y_probs[:, i]
        )
        avg_precision[i] = np.mean(precision[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(
        [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )
    save_path = os.path.join(settings.RESULTS_PATH, "pr_curves.png")

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"PR curve (class {class_names[i]}) (avg precision = {avg_precision[i]:.2f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return precision, recall


def __class_report(y_true, y_pred, class_names=None):
    """
    Generate and log the classification report.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        class_names (optional): List of class names for labeling the report.

    Returns:
        String representation of the classification report.
    """
    if class_names is None:
        report = metrics.classification_report(y_true, y_pred)
    else:
        report = metrics.classification_report(y_true, y_pred, target_names=class_names)

    logging.info("\nClassification Report:")
    logging.info(report)

    return report
