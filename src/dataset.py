import logging
import os
import time
from tqdm import tqdm

import torch
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
)  # , RandomHorizontalFlip, RandomCrop
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from . import settings


class Cifar10Dataset:
    def __init__(
        self,
    ):
        self.transform = self.__transform()

    def __transform(self):
        return Compose(
            [
                Resize(settings.IMG_SIZE),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Use ImageNet mean/std
            ]
        )

    def get_datasets(self):
        train_data = CIFAR10(
            root=settings.DATA_PATH, train=True, download=True, transform=self.transform
        )
        val_data = CIFAR10(
            root=settings.DATA_PATH,
            train=False,
            download=True,
            transform=self.transform,
        )
        class_names = train_data.classes
        return train_data, val_data, class_names

    @staticmethod
    def compute_mean_std(dataset, num_channels=3):
        loader = DataLoader(dataset, shuffle=False, num_workers=settings.NUM_WORKERS)
        before = time.time()
        mean = torch.zeros(num_channels)
        std = torch.zeros(num_channels)

        for inputs, _ in tqdm(loader, desc="Computing mean and std.."):
            for i in range(num_channels):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()

        mean.div_(len(dataset))
        std.div_(len(dataset))

        logging.info(f"Mean: {mean}, Std: {std}")
        logging.info(f"Time elapsed: {time.time() - before}")

        return mean, std


class CIFAR10EDA:
    @staticmethod
    def plot_class_distribution_and_pie_chart(dataset, class_names):
        labels = [label for _, label in dataset]
        class_counts = Counter(labels)
        counts = [class_counts[i] for i in range(len(class_names))]

        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar Plot
        sns.barplot(
            x=class_names,
            y=counts,
            ax=axes[0],
            hue=class_names,
            legend=False,
            palette="viridis",
        )
        axes[0].set_xticklabels(class_names, rotation=45)
        axes[0].set_title("Class Distribution in CIFAR-10")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        # Pie Chart
        axes[1].pie(
            counts,
            labels=class_names,
            autopct="%1.1f%%",
            colors=sns.color_palette("viridis", len(class_names)),
        )
        axes[1].set_title("Class Distribution in CIFAR-10")

        plt.tight_layout()
        plt.show()
