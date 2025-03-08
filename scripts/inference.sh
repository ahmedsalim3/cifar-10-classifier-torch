#!/bin/bash

export DISPLAY=:0

# python3 -m src.inference --model cnn --image_path "/path/to/local/image.jpg"
python3 -m src.inference --model cnn --image_path "https://media.istockphoto.com/id/521697371/photo/brown-pedigree-horse.jpg?s=612x612&w=0&k=20&c=x19W0K7iuQhQn_7l3wRqWq-zsbo0oRA33C3OF4nooL0=" --requests