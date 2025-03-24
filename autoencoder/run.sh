#!/bin/bash

source /speech/arjun/espnet-v.202409/tools/activate_python.sh

python train.py \
    --config_path /speech/arjun/exps/1study/generative_models/autoencoder/config.yaml \
