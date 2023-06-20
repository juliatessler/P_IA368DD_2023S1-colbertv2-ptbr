#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_CACHE="cache/"

conda activate colbert
python3 train.py
