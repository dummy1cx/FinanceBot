#!/bin/bash

# Create virtual environment
python3 -m venv venv

source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install torch torchvision matplotlib tqdm

pip freeze > requirements.txt

echo "Setup"

export WANDB_ENTITY=
export WANDB_PROJECT=
export USE_WANDB=true

echo "  Entity:   $WANDB_ENTITY"
echo "  Project:  $WANDB_PROJECT"

"
