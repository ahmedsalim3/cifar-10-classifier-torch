#!/bin/bash

python3 -m src.train_model --model cnn --epochs 30 --batch_size 64
python3 -m src.train_model --model vgg16 --epochs 30 --batch_size 64