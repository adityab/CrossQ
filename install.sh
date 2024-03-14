conda create -n crossq python=3.11.5
conda install -c nvidia cuda-nvcc

python -m pip install -e .
python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
