# CIFAR-10-Classifier

This repository contains a machine learning project for classifying [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) images using CNN and VGG16 models. It includes code for model training, evaluation, and inference, as well as example scripts and pretrained model weights.

## How to Install

Install required dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

### Training a Model

To train a model, run:

```bash
# Usage
python3 -m src.train_model [-h] --model {vgg16,cnn} [--epochs EPOCHS] [--batch_size BATCH_SIZE]
```

Example with transfer learning:

```bash
python3 -m src.train_model --model vgg16 --epochs 30 --batch_size 64
```

### Running Inference

To run inference on a trained model:

```bash
# Usage
python3 -m src.inference [-h] --model {cnn,vgg} --image_path IMAGE_PATH [--requests]
```

Use the `--requests` flag when providing a URL as the image path:
    
```sh
python3 -m src.inference --model cnn --image_path "https://media.istockphoto.com/id/521697371/photo/brown-pedigree-horse.jpg?s=612x612&w=0&k=20&c=x19W0K7iuQhQn_7l3wRqWq-zsbo0oRA33C3OF4nooL0=" --requests
```

**Note:** Due to the large size of the VGG16 model, we haven't included the weights in this repo, but we've included the [CNN weights](./models/cnn_classifier.pth).

## Repository Structure

The directory structure below shows the organization of files/directories in this repo:

```sh
template_repo
.
├── README.md
├── requirements.txt
│ 
├── data
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
│ 
├── models
│   ├── cnn_classifier.pth
│   └── vgg16_classifier.pth
│ 
├── notebook
│   └── inference.ipynb
│ 
├── results
│   ├── cifar_samples.png
│   ├── cnn
│   └── vgg
│ 
├── scripts
│   ├── inference.sh
│   └── train.sh
│
└── src
    ├── dataset.py
    ├── inference.py
    ├── models.py
    ├── settings.py
    ├── train_model.py
    └── utils/
```
