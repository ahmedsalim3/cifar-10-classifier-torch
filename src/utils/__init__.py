from .eval import *
from .infer import *

import matplotlib.pyplot as plt

def plot_images(data, num_imgs=4):
    fig, axes = plt.subplots(1, num_imgs, figsize=(15, num_imgs))

    for i in range(num_imgs):
        img, label = data[i]
        img = img.permute(1, 2, 0).numpy()

        img = (img - img.min()) / (img.max() - img.min())

        axes[i].imshow(img)
        axes[i].set_title(data.classes[label])
        axes[i].axis("off")

    plt.tight_layout()
    return fig
