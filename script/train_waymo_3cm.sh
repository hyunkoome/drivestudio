#!/bin/bash


export PYTHONPATH=/home/hyunkoo/DATA/ssd8tb/Journal/drivestudio:$PYTHONPATH

export $(cat .env | xargs) && python tools/train.py \
    --enable_wandb --entity hyunkookim-me \
    --config_file configs/omnire.yaml \
    --output_root output \
    --project drivestudio \
    --run_name omnire_waymo_3cm_scene_23 \
    dataset=waymo/3cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=-1