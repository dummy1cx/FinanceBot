#!/bin/bash

python3 -m venv venv

source venv/bin/activate


pip install --upgrade pip
pip install torch torchvision matplotlib tqdm

pip freeze > requirements.txt

echo "Setup"

export WANDB_ENTITY=
export WANDB_PROJECT="transformers_encoder_decoder_talking_head_90"
export USE_WANDB=true

echo "  Entity:   $WANDB_ENTITY"
echo "  Project:  $WANDB_PROJECT"
