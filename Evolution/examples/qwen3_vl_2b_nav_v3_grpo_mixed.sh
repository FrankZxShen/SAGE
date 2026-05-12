#!/bin/bash

set -x

export WANDB_API_KEY="xxx"
MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct  # replace it with your local file path
EXPERIMENT_NAME=Qwen3_vl_2b_nav_mix_data
SAVE_PATH=/path/to/${EXPERIMENT_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config_low_mem_mixed.yaml \
    data.train_files=/path/to/InteriorGS:0.5,/path/to/HM3D:0.5 \
    data.val_files=/path/to/InteriorGS:0.5,/path/to/HM3D:0.5 \
    data.usage_ratio=1 \
    data.val_ratio=0.2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.n_gpus_per_node=2
