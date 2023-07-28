# deepspeed env on A6000 48G GPU

python=3.8

wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sh cuda_11.3.0_465.19.01_linux.run

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

ds_report

python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
