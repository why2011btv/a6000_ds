# deepspeed env on A6000 48G GPU

## python
python=3.8
```
conda create --prefix ./a6000 python=3.8 pytorch cudatoolkit=11.1 -c pytorch -c nvidia 
```
(I used the wrong cuda version at first)

## cuda
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sh cuda_11.3.0_465.19.01_linux.run
```
(note: ONLY install CUDA Toolkit 11.3, no need to install Driver, CUDA Samples 11.3, CUDA Demo Suite 11.3, CUDA Documentation 11.3.
Go to Options -> Toolkit Options, Change Toolkit Install Path ==> /shared/why16gzl/Downloads/Downloads/cuda_113, unselect all other stuff
Probably need to go to Options -> Library install path (Blank for system default), and change it to the same custom path. Not sure about this.)

## ds_report
```
[2023-07-28 01:28:17,861] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-devel package with yum
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
No CUDA runtime is found, using CUDA_HOME='/shared/why16gzl/Downloads/Downloads/cuda_113'
DeepSpeed general environment info:
torch install path ............... ['/mnt/cogcomp-archive/shared/why16gzl/conda/miniconda38/envs/a6000/lib/python3.8/site-packages/torch']
torch version .................... 1.12.1+cu113
deepspeed install path ........... ['/mnt/cogcomp-archive/shared/why16gzl/conda/miniconda38/envs/a6000/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.10.0, unknown, unknown
torch cuda version ............... 11.3
torch hip version ................ None
nvcc version ..................... 11.3
deepspeed wheel compiled w. ...... torch 1.12, cuda 11.3
```

## torch
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"

[OUTPUT]:
1.12.1+cu113
[]
```