## UV Installation
First, install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Note:** You may need to set `CUDA_HOME` and have it pointing to a valid CUDA 12.x installation. To use a different CUDA version, please change the sources in `pyproject.toml` (and the `torch`/`torchvision` versions). See [this guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for more details.

Next, run:
```bash
uv sync
uv add --no-build-isolation \
'torch-scatter==2.1.2' \
'flash-attn~=2.7.4' \
'git+https://github.com/facebookresearch/detectron2.git@9604f5995cc628619f0e4fd913453b4d7d61db3f' \
'git+https://github.com/facebookresearch/pytorch3d.git@7a3c0cbc9d7b0e70ef39b7f3c35e9ce2b7376f32'
uv run bash docs/init.sh
```


## Conda Installation
Note: If you are using micromamba, set:
```
alias conda='micromamba'
```

Note: The instructions below install a standalone CUDA installation in your conda enviorment instead of using the system installation. You may need to use a different CUDA version based on your system drivers, GPU, etc. You can skip this step and instead just set `CUDA_HOME`, e.g., `export CUDA_HOME='/usr/local/cuda-12.4'`.

```
conda create -n univlg python=3.10
conda activate univlg
conda install cuda cuda-nvcc -c nvidia/label/cuda-12.4.1 # Optional
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH" # Optional
export CUDA_HOME=$CONDA_PREFIX # Optional
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt
bash docs/init.sh
```

### NTLK (Optional)
```
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("stopwords")'
```

## Troubleshooting

To support multiple GPU architectures, you will need to set `TORCH_CUDA_ARCH_LIST`. For example:
```
export TORCH_CUDA_ARCH_LIST="8.0 8.6"
```

See this [guide](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for more details. 