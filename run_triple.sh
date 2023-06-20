#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_CACHE="cache/"

python3 generate_dataset.py
