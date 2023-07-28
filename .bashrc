
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/shared/why16gzl/conda/miniconda38/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/shared/why16gzl/conda/miniconda38/etc/profile.d/conda.sh" ]; then
        . "/shared/why16gzl/conda/miniconda38/etc/profile.d/conda.sh"
    else
        export PATH="/shared/why16gzl/conda/miniconda38/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#cuda-9.2
#export PATH=/shared/why16gzl/cuda/cuda-9.2/bin:$PATH
#export LD_LIBRARY_PATH=/shared/why16gzl/cuda/cuda-9.2/lib64
#export PATH="$PATH:$HOME/bin"

#cuda-9.0
#export PATH=/shared/why16gzl/cuda-9.0/bin${PATH:+:${PATH}} # new
#export LD_LIBRARY_PATH=/usr/local/lib:/shared/why16gzl/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/lib # new
#export CUDA_HOME=/shared/why16gzl/cuda-9.0  #new

#cuda-10.2
#export PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/bin:$PATH
#export LD_LIBRARY_PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/lib64
#export CUDA_HOME=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2

#cuda-11.3
export PATH=/shared/why16gzl/Downloads/Downloads/cuda_113/bin:$PATH
export LD_LIBRARY_PATH=/shared/why16gzl/Downloads/Downloads/cuda_113/lib64
export CUDA_HOME=/shared/why16gzl/Downloads/Downloads/cuda_113

#huggingface
export HUGGINGFACE_HUB_CACHE=/shared/why16gzl/cache/huggingface/hub/

#torch
export TORCH_EXTENSIONS_DIR=/shared/why16gzl/cache/torch_extensions/py38_cu102 # for deepspeed
export TORCH_HOME=/shared/why16gzl/cache/torch/hub/

#maven
export M2_HOME=/shared/why16gzl/Downloads/apache-maven-3.6.3
export M2=$M2_HOME/bin
export MAVEN_OPTS=-Xms256m
export PATH=$M2:$PATH 

#java
export JAVA_HOME=/shared/why16gzl/Downloads/jdk-14.0.2
export PATH=$PATH:/shared/why16gzl/Downloads/jdk-14.0.2/bin

#Gurobi
export GUROBI_HOME="/shared/why16gzl/Downloads/gurobi811/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE=/home1/w/why16gzl/gurobi.lic

#docker
export PATH=/home1/w/why16gzl/bin:$PATH
export DOCKER_HOST=unix:///run/user/56512/docker.sock
