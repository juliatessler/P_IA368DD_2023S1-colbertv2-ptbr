# ColBERT-v2 PT-BR

[Júlia Tessler](https://github.com/juliatessler) and [Manoel Veríssimo](https://github.com/verissimomanoel)

This repository contains the code for the final project of IA-368 (Deep Learning for Information Retrieval) of Unicamp (University of Campinas) taken during the first semester of 2023.

## Set-up

We strongly suggest you create and activate a virtual environment to run this project. This can be achieved with:

```
python -m venv /path/to/venv
source /path/to/venv/bin/activate
```

Then, install the requirements:

```
pip install -r requirements.txt
```

You may not be able to run most of this code without a CUDA device.

---

## Usage

### Generating triples with distillation

You'll need a BM25 index. This can be achieved by following the steps from [mMARCO](https://github.com/unicamp-dl/mMARCO#bm25-baseline-for-portuguese).
