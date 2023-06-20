FROM nvcr.io/nvidia/pytorch:22.03-py3

# Install miniconda ----------------
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.9 pip wget

RUN conda install -c conda-forge openjdk=11 -y
RUN conda install -c pytorch faiss-gpu

WORKDIR /work

COPY requirements.txt /work/
COPY generate_dataset.py /work/
COPY run_triple.sh /work/

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN pip install -r /work/requirements.txt

ENTRYPOINT [ "./run_triple.sh" ]
