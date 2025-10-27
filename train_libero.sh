#!/bin/bash


MASTER_ADDR=${PET_MASTER_ADDR} 
MASTER_PORT=${PET_MASTER_PORT}

NUM_NODES=${PET_NNODES}  
GPUS_PER_NODE=${PET_NPROC_PER_NODE}
NODE_RANK=${PET_NODE_RANK}  

cd /HOME/ROOT_DIR

export WANDB_API_KEY='WANDB_KEY'


accelerate launch  --multi_gpu --num_machines ${PET_NNODES} --num_processes $(( ${PET_NNODES} * ${PET_NPROC_PER_NODE} )) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${PET_MASTER_ADDR} --main_process_port ${PET_MASTER_PORT} \
  --mixed_precision bf16 \
  train_omni_libero.py --output_dir 'OUTPUT_DIR' --resume_from_checkpoint 'RESUME_DIR' \
  --data_root_dir 'DATA_DIR' --max_train_steps 150000 --data_mix 'omniaction_libero_spatial' --checkpoint_save_frequency 100 --future_action_window_size 5 --shuffle_buffer_size 16000 --per_device_batch_size 4 --gradient_accumulation_steps 4 --learning_rate 5e-5 --logging_frequency 10 