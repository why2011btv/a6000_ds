#!/bin/bash
#SBATCH --job-name=why
#SBATCH --partition=p_nlp
#SBATCH --constraint=48GBgpu
#SBATCH --gpus=1
#SBATCH --mem=20GB
#SBATCH --nodelist=nlpgpu05
#SBATCH --output="/shared/why16gzl/Repositories/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/exp-%j.out"

echo $exp
hostnamectl
nvidia-smi
ps -ef |grep why
export HUGGINGFACE_HUB_CACHE=/shared/why16gzl/cache/huggingface/hub/
export PATH=/home1/w/why16gzl/bin:/shared/why16gzl/Downloads/apache-maven-3.6.3/bin:/shared/why16gzl/Downloads/Downloads/cuda_113/bin:/home1/w/why16gzl/bin:/shared/why16gzl/Downloads/apache-maven-3.6.3/bin:/mnt/cogcomp-archive/shared/why16gzl/conda/miniconda38/envs/a6000/bin:/shared/why16gzl/conda/miniconda38/condabin:/opt/seas/bin:/home1/w/why16gzl/.local/bin:/home1/w/why16gzl/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/mit/bin:/shared/why16gzl/Downloads/jdk-14.0.2/bin:/shared/why16gzl/Downloads/gurobi811/linux64/bin:.:/shared/why16gzl/Downloads/jdk-14.0.2/bin:/shared/why16gzl/Downloads/gurobi811/linux64/bin:.

cd /shared/why16gzl/Repositories/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/

OUTPUT_PATH=./output_lora
mkdir -p $OUTPUT_PATH

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-13b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
