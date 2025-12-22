# Installation

We recommend using [Conda](https://www.anaconda.com/) to create a Python environment for using Full-DIA, whether on Windows or Linux.

1. Create a Python environment with version 3.9.18.
    ```bash
    conda create -n full_env python=3.12
    conda activate full_env
    ```

2. Install the corresponding PyTorch and CuPy packages based on your CUDA version (which can be checked using the `nvidia-smi` command). Full-DIA requires an NVIDIA GPU with more than 10 GB of VRAM, a minimum of 64 GB RAM, and a high-performance Intel CPU.
  - CUDA-12
    ```bash
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    conda install cudatoolkit
    ```
  - CUDA-11
    ```bash
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    conda install cudatoolkit
    ```

3. Install Full-DIA
    ```bash
    pip install full_dia[cuda11] or pip install full_dia[cuda12]
    ```

- Alternatively, you can create a Conda environment with Full-DIA in one command:
    ```bash
    conda env create -f https://raw.githubusercontent.com/xomicsdatascience/full_dia/main/requirements/fulldia_cuda12.yml
    ```